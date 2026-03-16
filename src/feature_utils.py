"""
feature_utils.py
────────────────
Shared feature engineering utilities for the User Journey Funnel Predictor.

Imported by:
  - model_training.ipynb  (training)
  - app.py                (Flask inference)

Keeping all feature logic in one place ensures that the features produced
at inference time are byte-for-byte identical to those produced at training time.
If they diverge, predictions will be silently wrong — a classic production ML bug.
"""

import pandas as pd
from collections import Counter


# ── Constants ──────────────────────────────────────────────────────────────────

# Pages whose presence directly encodes the funnel stage label.
# These are excluded from X to prevent data leakage.
#   Checkout            → Signals 1, 2, 3
#   *certificate pages  → Signal 4
LEAKY_PAGES = {'Checkout', 'Career track certificate', 'Course certificate'}

# Funnel stage index → human-readable label
FUNNEL_STAGE_NAMES = {
    0: 'Browsing',
    1: 'Abandoned',
    2: 'Interested',
    3: 'Converted',
}

# Business recommendation per stage — returned by the Flask API
STAGE_RECOMMENDATIONS = {
    0: 'Show discovery content and awareness campaigns',
    1: 'Trigger cart-recovery email or retargeting ad',
    2: 'Offer a limited-time discount or free trial nudge',
    3: 'Upsell to annual plan or surface cross-sell content',
}

# Canonical page list (derived from training data, fixed for inference).
# Must match the order used when the model was trained.
# Update this if the page taxonomy changes and the model is retrained.
ALL_PAGES_ORDERED = [
    'About us', 'Blog', 'Career track certificate', 'Career tracks',
    'Checkout', 'Coupon', 'Course certificate', 'Courses', 'Homepage',
    'Instructors', 'Log in', 'Other', 'Pricing', 'Resources center',
    'Sign up', 'Success stories', 'Upcoming courses',
]

SAFE_PAGES = [p for p in ALL_PAGES_ORDERED if p not in LEAKY_PAGES]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _col(page: str) -> str:
    """Normalise a page name into a safe DataFrame column-name fragment."""
    return page.lower().replace(' ', '_').replace('-', '_')


# ── Preprocessing ──────────────────────────────────────────────────────────────

def remove_page_duplicates(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Collapse consecutive duplicate pages in each journey string.

    Example:
        'Log in-Log in-Log in-Pricing' → 'Log in-Pricing'

    Without this step, visit-count features are dominated by page-reload noise
    rather than genuine navigation behaviour.
    """
    def _clean(journey: str) -> str:
        if pd.isna(journey):
            return journey
        pages   = journey.split('-')
        cleaned = [pages[0]]
        for page in pages[1:]:
            if page != cleaned[-1]:
                cleaned.append(page)
        return '-'.join(cleaned)

    result = data.copy()
    result[target_column] = result[target_column].apply(_clean)
    return result


def group_by(
    data: pd.DataFrame,
    group_column:  str = 'user_id',
    target_column: str = 'user_journey',
    sessions:      object = 'All',
    count_from:    str = 'last',
) -> pd.DataFrame:
    """
    Concatenate all session-level journey strings into one per user.

    Args:
        sessions   : 'All' or int — how many sessions per user to include
        count_from : 'last' or 'first' — which sessions to keep when limiting
    """
    df = data.copy()
    if 'session_id' in df.columns:
        df = df.sort_values([group_column, 'session_id'])
    else:
        df = df.sort_values(group_column)

    if sessions != 'All' and isinstance(sessions, int):
        if count_from == 'last':
            df = df.groupby(group_column).tail(sessions)
        elif count_from == 'first':
            df = df.groupby(group_column).head(sessions)
        else:
            raise ValueError("count_from must be 'first' or 'last'")

    grouped = (
        df.groupby(group_column)[target_column]
        .apply(lambda x: '-'.join(x.astype(str)))
        .reset_index()
    )
    return grouped


# ── Labeling ───────────────────────────────────────────────────────────────────

def label_funnel_stage(data: pd.DataFrame, target_column: str = 'user_journey') -> pd.DataFrame:
    """
    Assign each user to a funnel stage using four independent conversion signals.

    Each signal contributes +1 to a composite score (0–4).
    Scores 3 and 4 are collapsed into Stage 3 (score-4 has only ~26 users).

    Signals:
        1. Visited Checkout at any point          (weak — includes abandoners)
        2. Visited Checkout AND Coupon            (medium — inside purchase flow)
        3. Journey exits on Checkout              (medium — session ended at purchase)
        4. Earned a certificate                   (strong — paid AND used product)

    Stages:
        0 → Browsing    (score 0)
        1 → Abandoned   (score 1)
        2 → Interested  (score 2)
        3 → Converted   (score 3–4)
    """
    df = data.copy()
    j  = df[target_column]

    sig1 = j.str.contains('Checkout', case=False, na=False).astype(int)
    sig2 = (
        j.str.contains('Checkout', case=False, na=False) &
        j.str.contains('Coupon',   case=False, na=False)
    ).astype(int)
    sig3 = j.str.endswith('Checkout').astype(int)
    sig4 = (
        j.str.contains('Career track certificate', case=False, na=False) |
        j.str.contains('Course certificate',       case=False, na=False)
    ).astype(int)

    df['conversion_score'] = sig1 + sig2 + sig3 + sig4
    df['funnel_stage']     = df['conversion_score'].apply(lambda s: min(s, 3))
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(
    data:          pd.DataFrame,
    target_column: str = 'user_journey',
) -> pd.DataFrame:
    """
    Transform raw user journey strings into a structured numeric feature matrix.

    Excludes pages in LEAKY_PAGES to prevent data leakage into the funnel label.
    Uses SAFE_PAGES (a fixed, sorted list) to guarantee consistent column ordering
    between training and inference — critical for correct predictions.

    Feature groups (67 total):
        Journey stats    (3)  : length, unique page count, session density
        Page presence   (14)  : binary visit flag per safe page
        Page counts     (14)  : visit frequency per safe page
        Entry page      (14)  : one-hot first page
        Exit page       (14)  : one-hot last page (leaky pages excluded)
        Sequence/signals (6)  : ordering and engagement patterns
        Subscription     (2)  : Annual / Monthly (Quarterly = baseline)

    Args:
        data          : DataFrame with user_journey (+ optional subscription_type)
        target_column : column holding the journey string

    Returns:
        pd.DataFrame  : one row per user, one column per feature, no label column
    """
    df = data.copy()

    feature_rows = []

    for _, row in df.iterrows():
        journey = row[target_column]
        pages   = journey.split('-') if pd.notna(journey) and journey else []
        counts  = Counter(pages)
        feat    = {}

        # ── 1. Journey-level statistics ───────────────────────────────────────
        feat['journey_length']    = len(pages)
        feat['unique_page_count'] = len(set(pages))
        feat['session_density']   = (
            feat['unique_page_count'] / feat['journey_length']
            if feat['journey_length'] > 0 else 0
        )

        # ── 2. Page presence flags — binary ───────────────────────────────────
        for page in SAFE_PAGES:
            feat[f'has_{_col(page)}'] = int(page in counts)

        # ── 3. Page visit counts ───────────────────────────────────────────────
        for page in SAFE_PAGES:
            feat[f'count_{_col(page)}'] = counts.get(page, 0)

        # ── 4. Entry page — one-hot ────────────────────────────────────────────
        entry_page = pages[0] if pages else None
        for page in SAFE_PAGES:
            feat[f'entry_{_col(page)}'] = int(entry_page == page)

        # ── 5. Exit page — one-hot (SAFE_PAGES excludes leaky pages) ──────────
        exit_page = pages[-1] if pages else None
        for page in SAFE_PAGES:
            feat[f'exit_{_col(page)}'] = int(exit_page == page)

        # ── 6. Sequence & behavioural signals ─────────────────────────────────
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

        # ── 7. Subscription type — one-hot (Quarterly = baseline) ─────────────
        if 'subscription_type' in row.index:
            sub = str(row['subscription_type']).lower()
            feat['sub_annual']  = int(sub == 'annual')
            feat['sub_monthly'] = int(sub == 'monthly')
        else:
            # Inference default when subscription_type is not provided
            feat['sub_annual']  = 0
            feat['sub_monthly'] = 0

        feature_rows.append(feat)

    return pd.DataFrame(feature_rows)


def engineer_single(journey: str, subscription_type: str = 'Quarterly') -> pd.DataFrame:
    """
    Convenience wrapper for single-user inference in the Flask API.

    Args:
        journey           : raw journey string, e.g. 'Homepage-Pricing-Sign up'
        subscription_type : 'Annual', 'Monthly', or 'Quarterly'

    Returns:
        pd.DataFrame with one row — ready to pass directly to model.predict()
    """
    row_df = pd.DataFrame([{
        'user_journey':      journey,
        'subscription_type': subscription_type,
    }])
    return engineer_features(row_df)


def get_funnel_matrix(
    labeled_data:  pd.DataFrame,
    target_column: str = 'user_journey',
    label_column:  str = 'funnel_stage',
):
    """
    Build X (feature matrix) and y (labels) for model training.

    Returns:
        X             : pd.DataFrame
        y             : pd.Series
        feature_names : list[str]
    """
    X = engineer_features(labeled_data, target_column=target_column)
    y = labeled_data[label_column].reset_index(drop=True)
    return X, y, X.columns.tolist()
