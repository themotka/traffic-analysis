from .base import Handler


class NormalizeColumnsHandler(Handler):
    def process(self, df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(",", "")
        )
        return df
