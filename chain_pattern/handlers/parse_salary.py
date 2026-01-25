import re
import numpy as np
from .base import Handler


class ParseSalaryHandler(Handler):
    def process(self, df):
        def parse_salary(value: str) -> int:
            digits = re.sub(r"[^\d]", "", value)
            return int(digits) if digits else 0

        df["salary"] = df["зп"].apply(parse_salary)
        return df