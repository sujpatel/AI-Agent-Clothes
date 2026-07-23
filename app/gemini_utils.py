import logging

from google.genai import errors
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Retries transient failures (mainly 503 "model overloaded", which we've hit
# for real) with exponential backoff, instead of letting them crash the request.
with_gemini_retry = retry(
    retry=retry_if_exception_type(errors.ServerError),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(4),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
