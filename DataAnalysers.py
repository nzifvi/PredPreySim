from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy
import pandas

@dataclass
class MessageLogProcessor:
    df : pandas.DataFrame
    requiredColumns = {
        "step",
        "species",
        "genotypeID",
        "agentIndex",
        "x",
        "y",
        "vx",
        "vy",
        "alive",
        "msg0",
        "msg1",
        "receivedMessage0",
        "receivedMessage1",
        "move0",
        "move1",
        "nearestEnemyDist",
        "nearestAllyDist",
        "visibleEnemyCount",
        "visibleAllyCount"
    }

    @classmethod
    def fromCsv(cls, path:str | Path) -> "MessageLogProcessor":
        path = Path(path)
        df = pandas.read_csv(path)
        processor = cls(df)
        processor._validateColumns()
        processor._addDerivedColumns()
        return processor

    def _validateColumns(self) -> None:
        missing = self.requiredColumns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _addDerivedColumns(self) -> None:
        self.df["messageMagnitude"] = numpy.sqrt(
            self.df["msg0"] ** 2 + self.df["msg1"] ** 2
        )
        self.df["receivedMessageMagnitude"] = numpy.sqrt(
            self.df["receivedMessage0"] ** 2 + self.df["receivedMessage1"] ** 2
        )
        self.df["speed"] = numpy.sqrt(
            self.df["vx"] ** 2 + self.df["vy"] ** 2
        )
        self.df["moveMagnitude"] = numpy.sqrt(
            self.df["move0"] ** 2 + self.df["move1"] ** 2
        )
        self.df["seesEnemy"] = self.df["visibleEnemyCount"] > 0
        self.df["seesAlly"] = self.df["visibleAllyCount"] > 0

    def copy(self) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df.copy())

    def species(self, species:str) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df[self.df["species"] == species].copy())

    def genotype(self, genotypeID:int) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df[(self.df["genotypeID"] == genotypeID)].copy())

    def stepRange(self, start:int, end:int)-> "MessageLogProcessor":
        mask = (self.df["step"] >= start) & (self.df["step"] <= end)
        return MessageLogProcessor(self.df[mask].copy())

    def aliveOnly(self) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df[self.df["alive"]].copy())

    def enemyVisibleOnly(self) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df[self.df["visibleEnemyCount"] > 0].copy())

    def allyVisibleOnly(self) -> "MessageLogProcessor":
        return MessageLogProcessor(self.df[self.df["visibleAllyCount"] > 0].copy())

    def summary(self) -> pandas.DataFrame:
        summaryDict = {
            "rows" : [len(self.df)],
            "speciesCount" : [self.df["species"].nunique()],
            "genotypeCount" : [self.df["genotypeID"].nunique()],
            "maxStep" : [self.df["step"].max()],
            "meanMsg0" : [self.df["msg0"].mean()],
            "meanMsg1" : [self.df["msg1"].mean()],
            "meanMessageMagnitude" : [self.df["receivedMessageMagnitude"].mean()],
            "meanReceivedMessageMagnitude" : [self.df["receivedMessageMagnitude"].mean()],
            "meanSpeed" : [self.df["speed"].mean()],
            "meanVisibleEnemyCount" : [self.df["visibleEnemyCount"].mean()],
            "meanVisibleAllyCount" : [self.df["visibleAllyCount"].mean()],
        }
        return pandas.DataFrame(summaryDict)

    def perSpeciesSummary(self) -> pandas.DataFrame:
        return (
            self.df.groupby("species", as_index=False).agg(
                rows=("species", "size"),
                meanMsg0 = ("msg0", "mean"),
                meanMsg1 = ("msg1", "mean"),
                meanMessageMagnitude = ("msg0", "mean"),
                meanReceivedMessageMagnitude = ("msg0", "mean"),
                meanSpeed = ("speed", "mean"),
                meanVisibleEnemyCount = ("visibleEnemyCount", "mean"),
                meanVisibleAllyCount = ("visibleAllyCount", "mean"),
            ).sort_values("rows", ascending = False)
        )

    def perGenotypeSummary(self, species:Optional[str] = None) -> pandas.DataFrame:
        df = self.df if species is None else self.df[self.df["species"] == species]

        return (
            df.groupby(["species", "genotypeID"], as_index=False).agg(
                rows = ("genotypeID", "size"),
                meanMsg0 = ("msg0", "mean"),
                meanMsg1 = ("msg1", "mean"),
                stdMsg0 = ("msg0", "std"),
                stdMsg1 = ("msg1", "std"),
                meanMessageMagnitude = ("msg0", "mean"),
                meanReceivedMessageMagnitude = ("msg0", "mean"),
                meanSpeed = ("speed", "mean"),
                meanMoveMagnitude = ("moveMagnitude", "mean"),
                enemyVisibleRate = ("seesEnemy", "mean"),
                allyVisibleRate = ("seesAlly", "mean"),
                meanNearestEnemyDist = ("nearestEnemyDist", "mean"),
                meanNearestAllyDist = ("nearestAllyDist", "mean")
            ).sort_values(["species", "genotypeID"])
        )

    def contextMessageSummary(self) -> pandas.DataFrame:
        return (
            self.df.groupby(["species", "seesEnemy", "seesAlly"], as_index=False).agg(
                rows = ("species", "size"),
                meanMsg0 = ("msg0", "mean"),
                meanMsg1 = ("msg1", "mean"),
                meanMessageMagnitude = ("messageMagnitude", "mean"),
                meanReceivedMessageMagnitude = ("receivedMessageMagnitude", "mean"),
                meanMove0 = ("move0", "mean"),
                meanMove1 = ("move1", "mean"),
                meanSpeed = ("speed", "mean"),
            ).sort_values(["species", "seesEnemy", "seesAlly"])
        )

    def messageVariabilityByGenotype(self, species: Optional[str] = None) -> pandas.DataFrame:
        df = self.df if species is None else self.df[self.df["species"] == species]

        grouped = (
            df.groupby(["species", "genotypeID"], as_index=False).agg(
                stdMsg0 = ("msg0", "std"),
                stdMsg1 = ("msg1", "std"),
                messageMagnitudeMean = ("messageMagnitude", "mean"),
                messageMagnitudeStd = ("messageMagnitude", "std"),
                rows = ("genotypeID", "size")
            ).sort_values(["species", "messageMagnitudeStd"], ascending = [True, False])
        )
        return grouped

    def messageCorrelationTable(self) -> pandas.DataFrame:
        cols = [
            "msg0",
            "msg1",
            "receivedMessage0",
            "receivedMessage1",
            "move0",
            "move1",
            "nearestEnemyDist",
            "nearestAllyDist",
            "visibleEnemyCount",
            "visibleAllyCount",
            "speed",
            "messageMagnitude",
            "receivedMessageMagnitude"
        ]
        return self.df[cols].corr(numeric_only=True)

    def discretiseMessages(self, bins:int = 10) -> pandas.DataFrame:
        df = self.df.copy()

        df["msg0Bin"] = pandas.cut(
            df["msg0"],
            bins = bins,
            labels = False,
            include_lowest = True
        )
        df["msg1Bin"] = pandas.cut(
            df["msg1"],
            bins = bins,
            labels = False,
            include_lowest = True
        )

        counts = (
            df.groupby(["species", "msg0Bin", "msg1Bin"], as_index = False).size().rename(columns={"size":"count"}).sort_values("count", ascending=False)
        )
        return counts

    def topMessageBins(self, bins:int = 10, topN: int = 20) -> pandas.DataFrame:
        return self.discretiseMessages(bins = bins).head(topN)

