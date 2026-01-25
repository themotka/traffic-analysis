import re
from .base import Handler


class ParseGenderAgeHandler(Handler):
    def process(self, df):
        def parse_gender(value: str) -> int:
            return 1 if "муж" in value.lower() else 0

        def parse_age(value: str) -> int:
            match = re.search(r"(\d+)\s*года", value)
            return int(match.group(1)) if match else -1

        df["gender"] = df["пол_возраст"].apply(parse_gender)
        df["age"] = df["пол_возраст"].apply(parse_age)

        df.drop(columns=["пол_возраст"], inplace=True)
        return df
