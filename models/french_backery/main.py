import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class Row:
    date: str
    time: str
    ticket_number: int
    article: str
    quantity: str
    unit_price: str


@dataclass
class Backery:
    articles: list[Row]


df = pd.read_csv("Bakery sales.csv")
backery=Backery(articles=[])

for row in df[["date", "time", "ticket_number", "article", "Quantity", "unit_price"]].values:
    backery.articles.append(Row(
        date=row[0], time=row[1], ticket_number=int(row[2]),
        article=row[3], quantity=row[4],
        unit_price=float(row[5].split()[0].replace(',', '.'))
    ))


with open("backery.json", "w") as f:
    json.dump(asdict(backery), fp=f, indent=4)
