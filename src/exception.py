import sys #sys library will give info about the exception that has occurred in the runtime

from src.logger import logging #importing logging module from logger.py file
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #exc_info() method of sys module will give info about the exception that has occurred in the runtime
    file_name = exc_tb.tb_frame.f_code.co_filename #getting the file name where the exception has occurred
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error) #getting the line number where the exception has occurred
    )
    return error_message

class CustomException(Exception): #inheriting from Exception class
    def __init__(self, error, error_detail:sys): #constructor
        super().__init__(error) #calling the constructor of the parent class
        self.error = error
        self.error_detail = error_detail

    def __str__(self):
        return error_message_detail(self.error, self.error_detail) #returning the error message with details
    