import sys
from typing import Optional, Tuple, Type


def build_error_message(error: Exception, error_detail_module=sys) -> str:
    """
    Return a readable error message with file and line of the original exception.
    Safe even if no traceback is present.
    """
    exc_type, exc_obj, exc_tb = error_detail_module.exc_info()  # type: Optional[Tuple[Type[BaseException], BaseException, object]]
    if exc_tb is None:
        # Fallback: no traceback available
        return f"{type(error).__name__}: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error in {file_name} at line {line_no}: {type(error).__name__}: {error}"


class CustomException(Exception):
    """
    Wraps an underlying exception and exposes a clean, human-readable message
    that includes file and line number.
    """
    def __init__(self, error: Exception):
        message = build_error_message(error, sys)
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
