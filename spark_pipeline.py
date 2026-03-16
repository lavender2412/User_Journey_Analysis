"""
spark_pipeline.py
─────────────────
PySpark ETL pipeline — User Journey Funnel Feature Engineering

Architecture position:

    Raw data (S3 / local CSV)
        │
        ▼  [this script]
    Processed feature table (Parquet / S3)
        │
        ▼
    model_training.ipynb  (sklearn — local dev or SageMaker)
        │
        ▼
    app.py  (Flask REST inference)

Why PySpark here?
─────────────────
In production, clickstream data lives in a data lake (S3, GCS, HDFS) as
billions of raw events, not a single CSV. Spark reads that data in parallel
across a cluster (AWS EMR, Databricks, GCP Dataproc) without loading everything
into memory on one machine. This script handles the heavy ETL — session
ordering, deduplication, journey aggregation, labeling, and feature engineering
— before sklearn ever sees any data.

Local dev:
    python spark_pipeline.py

Production (AWS EMR):
    spark-submit --master yarn \\
                 --deploy-mode cluster \\
                 --num-executors 10 \\
                 --executor-cores 4 \\
                 --executor-memory 8g \\
                 spark_pipeline.py \\
                 --input  s3://your-bucket/raw/user_journey_raw.csv \\
                 --output s3://your-bucket/features/funnel_features/
"""

import argparse
from collections import Counter

import pandas as pd
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)


# ── Feature constants — must mirror feature_utils.py EXACTLY ─────────────────
# Any divergence here means training and inference see different feature spaces,
# causing silent prediction errors.

LEAKY_PAGES = {'Checkout', 'Career track certificate', 'Course certificate'}

ALL_PAGES_ORDERED = [
    'About us', 'Blog', 'Career track certificate', 'Career tracks',
    'Checkout', 'Coupon', 'Course certificate', 'Courses', 'Homepage',
    'Instructors', 'Log in', 'Other', 'Pricing', 'Resources center',
    'Sign up', 'Success stories', 'Upcoming courses',
]

SAFE_PAGES = [p for p in ALL_PAGES_ORDERED if p not in LEAKY_PAGES]

FUNNEL_STAGE_NAMES = {
    0: 'Browsing',
    1: 'Abandoned',
    2: 'Interested',
    3: 'Converted',
}


def _col_name(page: str) -> str:
    """Normalise a page name into a safe column-name fragment."""
    return page.lower().replace(' ', '_').replace('-', '_')


# ── Output schema (metadata cols + 67 feature cols) ──────────────────────────
# Defining the schema upfront lets Spark validate the Pandas UDF output and
# avoids the overhead of schema inference on large datasets.

def _build_feature_schema() -> StructType:
    fields = [
        StructField('user_id',           StringType(),  False),
        StructField('subscription_type', StringType(),  True),
        StructField('user_journey',      StringType(),  True),
        StructField('funnel_stage',      IntegerType(), True),
        # ── Journey-level stats (3) ──────────────────────────────────────────
        StructField('journey_length',    IntegerType(), True),
        StructField('unique_page_count', IntegerType(), True),
        StructField('session_density',   DoubleType(),  True),
    ]
    # ── Page presence flags (14) ─────────────────────────────────────────────
    for page in SAFE_PAGES:
        fields.append(StructField(f'has_{_col_name(page)}',   IntegerType(), True))
    # ── Page visit counts (14) ───────────────────────────────────────────────
    for page in SAFE_PAGES:
        fields.append(StructField(f'count_{_col_name(page)}', IntegerType(), True))
    # ── Entry page one-hot (14) ──────────────────────────────────────────────
    for page in SAFE_PAGES:
        fields.append(StructField(f'entry_{_col_name(page)}', IntegerType(), True))
    # ── Exit page one-hot (14) ───────────────────────────────────────────────
    for page in SAFE_PAGES:
        fields.append(StructField(f'exit_{_col_name(page)}',  IntegerType(), True))
    # ── Sequence / behavioural signals (6) ───────────────────────────────────
    for name in [
        'signup_before_login', 'career_engaged',
        'visited_pricing',     'pricing_visit_count',
        'visited_coupon',      'coupon_visit_count',
    ]:
        fields.append(StructField(name, IntegerType(), True))
    # ── Subscription type one-hot (2) ─────────────────────────────────────────
    fields.append(StructField('sub_annual',  IntegerType(), True))
    fields.append(StructField('sub_monthly', IntegerType(), True))
    return StructType(fields)


FEATURE_SCHEMA = _build_feature_schema()


# ── UDF: remove consecutive duplicate pages ──────────────────────────────────

@F.udf(StringType())
def remove_consecutive_duplicates(journey: str) -> str:
    """
    Collapse consecutive duplicate pages in a journey string.

    Example:
        'Log in-Log in-Pricing-Pricing-Checkout' → 'Log in-Pricing-Checkout'

    Mirrors feature_utils.remove_page_duplicates() for Spark row-level use.
    Registered as a UDF so it runs on every executor without driver overhead.
    """
    if not journey:
        return journey
    pages   = journey.split('-')
    cleaned = [pages[0]]
    for page in pages[1:]:
        if page != cleaned[-1]:
            cleaned.append(page)
    return '-'.join(cleaned)


# ── UDF: join ordered session structs into a single journey string ────────────

@F.udf(StringType())
def concat_sessions(sessions) -> str:
    """
    Join a sort_array of (session_id, user_journey) structs into one string.

    Spark's collect_list does NOT guarantee order, so we use sort_array on a
    struct{session_id, user_journey} to preserve chronological order before
    joining. This is the Spark equivalent of:

        df.sort_values('session_id')
          .groupby('user_id')['user_journey']
          .apply('-'.join)
    """
    if not sessions:
        return ''
    parts = [s['user_journey'] for s in sessions if s['user_journey']]
    return '-'.join(parts)


# ── Pandas UDF: full feature engineering (partition-level, Arrow-accelerated) ─
# mapInPandas sends an entire partition to each executor as a pandas DataFrame.
# This is far more efficient than a row-level Python UDF because:
#   1. Arrow serialisation replaces slow pickle-based row-by-row serialisation.
#   2. We avoid Python interpreter overhead per row — one function call per partition.
# The tradeoff: we must declare the output schema upfront (FEATURE_SCHEMA above).

def _engineer_row_dict(journey: str, subscription: str) -> dict:
    """
    Pure-Python feature engineering for a single user.
    Called inside the Pandas UDF — runs on Spark executors, not the driver.
    Must stay in sync with feature_utils.engineer_features() at all times.
    """
    pages  = journey.split('-') if journey else []
    counts = Counter(pages)
    feat   = {}

    # Journey stats
    feat['journey_length']    = len(pages)
    feat['unique_page_count'] = len(set(pages))
    feat['session_density']   = (
        len(set(pages)) / len(pages) if pages else 0.0
    )

    # Page presence flags & visit counts
    for page in SAFE_PAGES:
        c = _col_name(page)
        feat[f'has_{c}']   = int(page in counts)
        feat[f'count_{c}'] = counts.get(page, 0)

    # Entry / exit one-hot
    entry_page = pages[0]  if pages else None
    exit_page  = pages[-1] if pages else None
    for page in SAFE_PAGES:
        c = _col_name(page)
        feat[f'entry_{c}'] = int(entry_page == page)
        feat[f'exit_{c}']  = int(exit_page  == page)

    # Sequence signals
    if 'Sign up' in counts and 'Log in' in counts:
        feat['signup_before_login'] = int(
            pages.index('Sign up') < pages.index('Log in')
        )
    else:
        feat['signup_before_login'] = 0

    feat['career_engaged']      = int('Career tracks' in counts or 'Courses' in counts)
    feat['visited_pricing']     = int('Pricing' in counts)
    feat['pricing_visit_count'] = counts.get('Pricing', 0)
    feat['visited_coupon']      = int('Coupon' in counts)
    feat['coupon_visit_count']  = counts.get('Coupon', 0)

    # Subscription type (Quarterly = baseline; both flags 0)
    sub = (subscription or '').lower()
    feat['sub_annual']  = int(sub == 'annual')
    feat['sub_monthly'] = int(sub == 'monthly')

    return feat


def _engineer_partition(iterator):
    """
    Partition-level function supplied to df.mapInPandas().

    mapInPandas passes an *iterator* of pandas DataFrames (one per internal
    batch), and expects the function to yield pandas DataFrames back.
    This is the correct signature for ALL PySpark versions (3.x and 4.x).

    Each yielded DataFrame must match FEATURE_SCHEMA exactly — column names,
    column order, and dtypes — or Spark will raise a schema mismatch error.
    """
    schema_cols = [field.name for field in FEATURE_SCHEMA.fields]

    for pdf in iterator:
        rows = []
        for _, row in pdf.iterrows():
            feat = _engineer_row_dict(
                journey      = row['user_journey'],
                subscription = row.get('subscription_type', 'Quarterly'),
            )
            # Prepend the metadata columns in schema order.
            # Belt-and-suspenders str() cast: even if the Spark ingestion cast
            # above is somehow bypassed, Arrow will never see an int64 user_id.
            record = {
                'user_id':           str(row['user_id']),
                'subscription_type': str(row['subscription_type']) if row.get('subscription_type') else 'Quarterly',
                'user_journey':      row['user_journey'],
                'funnel_stage':      int(row['funnel_stage']),
            }
            record.update(feat)
            rows.append(record)

        yield pd.DataFrame(rows, columns=schema_cols)


# ── SparkSession factory ──────────────────────────────────────────────────────

def build_spark_session(app_name: str = 'UserJourneyFunnelETL') -> SparkSession:
    """
    Create a SparkSession for local development.

    On a cluster (EMR, Databricks, Dataproc), SparkSession.builder picks up
    the cluster configuration automatically — the .master() and .config() calls
    below are silently overridden. No code changes are needed for production.

    Notable configs:
        arrow.pyspark.enabled      → use Apache Arrow for Pandas UDF serialisation
                                     (10-100× faster than pickle for mapInPandas)
        shuffle.partitions = 8     → sensible for small local datasets;
                                     increase to 200-400 for large cluster jobs
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master('local[*]')                        # all local cores; ignored on cluster
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.sql.shuffle.partitions', '8')
        .getOrCreate()
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path:  str = 'user_journey_raw.csv',
    output_path: str = 'spark_output/funnel_features',
    n_sessions:  str = 'all',
    count_from:  str = 'last',
) -> None:
    """
    Execute the full ETL pipeline.

    Steps:
        1. Ingest raw CSV (or S3 path in production)
        2. Order sessions chronologically per user
        3. Deduplicate consecutive pages (per session)
        4. Aggregate sessions → one journey string per user
        5. Assign funnel stage labels
        6. Engineer 67 numeric features via Pandas UDF
        7. Write partitioned Parquet output

    Args:
        input_path  : Path to raw CSV locally, or s3://bucket/key in production.
        output_path : Parquet destination — local directory or S3 prefix.
        n_sessions  : How many sessions per user to include ('all' or an integer).
        count_from  : Which sessions to keep when n_sessions is an integer
                      ('last' keeps most-recent; 'first' keeps oldest).
    """
    spark = build_spark_session()
    spark.sparkContext.setLogLevel('WARN')

    print('\n' + '=' * 62)
    print('  User Journey Funnel — PySpark ETL Pipeline')
    print('=' * 62)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    print(f'\n[1/6] Ingesting raw data')
    print(f'      Source : {input_path}')

    raw_df = (
        spark.read
        .option('header',      'true')
        .option('inferSchema', 'true')
        .csv(input_path)
        # inferSchema reads numeric-looking IDs as int64 but our schema needs string.
        # Cast here so the type contract is enforced before any downstream operation.
        .withColumn('user_id', F.col('user_id').cast('string'))
    )
    raw_df.cache()

    n_sessions_raw = raw_df.count()
    n_users_raw    = raw_df.select('user_id').distinct().count()
    print(f'      Rows   : {n_sessions_raw:,} sessions')
    print(f'      Users  : {n_users_raw:,} unique users')
    print(f'      Columns: {raw_df.columns}')

    # ── Step 2: Session ordering & optional per-user limiting ─────────────────
    print(f'\n[2/6] Ordering sessions')
    print(f'      n_sessions={n_sessions}, count_from={count_from}')

    # Window partitioned by user, ordered by session_id (ascending = chronological)
    user_window = Window.partitionBy('user_id').orderBy(F.col('session_id').cast('long'))
    session_df  = raw_df.withColumn('_session_rank', F.row_number().over(user_window))

    if n_sessions.lower() != 'all':
        n = int(n_sessions)
        if count_from == 'last':
            # Keep the n most-recent sessions: rank from the end using desc order
            rev_window = Window.partitionBy('user_id').orderBy(F.col('session_id').cast('long').desc())
            session_df = (
                session_df
                .withColumn('_rev_rank', F.row_number().over(rev_window))
                .filter(F.col('_rev_rank') <= n)
                .drop('_rev_rank')
            )
        else:  # 'first'
            session_df = session_df.filter(F.col('_session_rank') <= n)

    session_df = session_df.drop('_session_rank')
    n_after_limit = session_df.count()
    print(f'      {n_after_limit:,} sessions retained')

    # ── Step 3: Deduplicate consecutive pages (per-session) ───────────────────
    print('\n[3/6] Removing consecutive duplicate pages (per session)')

    deduped_df = session_df.withColumn(
        'user_journey',
        remove_consecutive_duplicates(F.col('user_journey'))
    )

    # ── Step 4: Aggregate sessions → one journey per user ────────────────────
    print('\n[4/6] Aggregating sessions → one journey string per user')
    #
    # Pattern: collect_list of structs → sort_array by session_id → concat journeys
    #
    # We collect a struct{session_id (as long for numeric sort), user_journey}
    # per user. sort_array sorts these structs lexicographically, but since the
    # first field is a long (session_id), the sort is chronological.
    # We then pass the sorted array to concat_sessions() to join the strings.
    #
    # This is the Spark equivalent of:
    #   df.sort_values(['user_id', 'session_id'])
    #     .groupby('user_id')['user_journey'].apply('-'.join)

    user_df = (
        deduped_df
        .withColumn(
            '_session_struct',
            F.struct(
                F.col('session_id').cast('long').alias('session_id'),
                F.col('user_journey').alias('user_journey'),
            )
        )
        .groupBy('user_id')
        .agg(
            F.sort_array(
                F.collect_list('_session_struct')
            ).alias('_sessions'),
            # Take the last non-null subscription_type per user
            F.last('subscription_type', ignorenulls=True).alias('subscription_type'),
        )
        .withColumn('user_journey', concat_sessions(F.col('_sessions')))
        .drop('_sessions')
    )

    # Cross-session dedup: adjacent pages at session boundaries may duplicate
    user_df = user_df.withColumn(
        'user_journey',
        remove_consecutive_duplicates(F.col('user_journey'))
    )

    user_df.cache()
    n_users_agg = user_df.count()
    print(f'      {n_users_agg:,} user-level rows after aggregation')

    # ── Step 5: Funnel stage labeling ─────────────────────────────────────────
    print('\n[5/6] Assigning funnel stage labels')
    #
    # Four independent signals, each contributing +1 to a composite score (0–4).
    # Scores ≥ 3 collapse to Stage 3 (Converted) via F.least(..., lit(3)).
    # This is a direct Spark translation of feature_utils.label_funnel_stage().
    #
    # Note: F.col().contains() and .endswith() are native Spark string functions —
    # no UDF needed here, so this runs at full JVM speed without Python overhead.

    sig1 = F.col('user_journey').contains('Checkout').cast('int')
    sig2 = (
        F.col('user_journey').contains('Checkout') &
        F.col('user_journey').contains('Coupon')
    ).cast('int')
    sig3 = F.col('user_journey').endswith('Checkout').cast('int')
    sig4 = (
        F.col('user_journey').contains('Career track certificate') |
        F.col('user_journey').contains('Course certificate')
    ).cast('int')

    labeled_df = (
        user_df
        .withColumn('_score',      sig1 + sig2 + sig3 + sig4)
        .withColumn('funnel_stage', F.least(F.col('_score'), F.lit(3)))
        .drop('_score')
    )

    # Print stage distribution as a sanity check
    print('\n      Stage distribution:')
    stage_rows = (
        labeled_df
        .groupBy('funnel_stage')
        .count()
        .orderBy('funnel_stage')
        .collect()
    )
    total = sum(r['count'] for r in stage_rows)
    for r in stage_rows:
        label = FUNNEL_STAGE_NAMES.get(r['funnel_stage'], '?')
        pct   = 100 * r['count'] / total if total else 0
        print(f'        Stage {r["funnel_stage"]} ({label:<10s}): {r["count"]:4d} users ({pct:.1f}%)')

    # ── Step 6: Feature engineering ───────────────────────────────────────────
    print('\n[6/6] Engineering features (mapInPandas — Arrow-accelerated)')
    #
    # mapInPandas sends each Spark partition to an executor as a pandas DataFrame,
    # applies _engineer_partition(), and collects the results back.
    #
    # Why mapInPandas over a row-level UDF?
    #   - Arrow serialisation: ~10-100× faster than pickle-based row UDFs.
    #   - Batched Python logic: Counter, list comprehensions run over a chunk,
    #     not one row at a time.
    #   - Exact schema enforcement: Spark validates the output against FEATURE_SCHEMA.

    feature_df = labeled_df.mapInPandas(_engineer_partition, schema=FEATURE_SCHEMA)
    feature_df.cache()

    n_feature_cols = len(FEATURE_SCHEMA.fields) - 4  # minus user_id, sub, journey, label
    n_rows         = feature_df.count()
    print(f'      Feature columns : {n_feature_cols}')
    print(f'      Output rows     : {n_rows:,}')

    # ── Write Parquet ─────────────────────────────────────────────────────────
    print(f'\n[Output] Writing Parquet')
    print(f'         Destination  : {output_path}')
    print(f'         Partitioned by: funnel_stage  (enables partition pruning)')

    (
        feature_df
        .repartition('funnel_stage')   # one file-set per stage
        .write
        .mode('overwrite')
        .partitionBy('funnel_stage')
        .parquet(output_path)
    )

    print('\n  Pipeline complete. Feature table written successfully.')
    print('\n  To load in model_training.ipynb (Spark reader):')
    print(f"    df = spark.read.parquet('{output_path}')")
    print(f"    X, y = df.drop('funnel_stage').toPandas(), df.select('funnel_stage').toPandas()")
    print('\n  Or load the Parquet directly with pandas:')
    print(f"    import pandas as pd")
    print(f"    df = pd.read_parquet('{output_path}')")

    # ── Sanity-check sample ───────────────────────────────────────────────────
    print('\n[Sample rows]:')
    feature_df.select(
        'user_id', 'funnel_stage', 'journey_length', 'unique_page_count',
        'visited_pricing', 'visited_coupon', 'sub_annual', 'sub_monthly',
    ).show(5, truncate=False)

    print('=' * 62 + '\n')
    spark.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='User Journey Funnel — PySpark ETL Pipeline'
    )
    parser.add_argument(
        '--input',
        default='user_journey_raw.csv',
        help="Path to raw CSV (local) or s3://bucket/key (production)."
             " Default: user_journey_raw.csv",
    )
    parser.add_argument(
        '--output',
        default='spark_output/funnel_features',
        help="Parquet output path (local dir) or s3://bucket/prefix."
             " Default: spark_output/funnel_features",
    )
    parser.add_argument(
        '--n-sessions',
        default='all',
        help="Sessions per user to include: 'all' or an integer. Default: all",
    )
    parser.add_argument(
        '--count-from',
        default='last',
        choices=['last', 'first'],
        help="When n-sessions is an integer, keep 'last' (most recent) or"
             " 'first' (oldest) sessions. Default: last",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path  = args.input,
        output_path = args.output,
        n_sessions  = args.n_sessions,
        count_from  = args.count_from,
    )
