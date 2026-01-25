from .base import Handler


class ParseCityHandler(Handler):
    def process(self, df):
        df["city"] = df["город"].apply(lambda x: x.split(",")[0].strip())
        df.drop(columns=["город"], inplace=True)
        return df
