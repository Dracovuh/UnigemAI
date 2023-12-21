from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    class Postgres:
        user = os.getenv("PostgresUser")
        database = os.getenv("PostgresDatabase")
        password = os.getenv("PostgresPassword")
        port = os.getenv("PostgresPort")
        host = os.getenv("PostgresHost")
        schema = os.getenv("PostgresSchema")

    Port = os.getenv("PORT")

    class Training:
        WS = 12

class Feature:
    Inputs = [

    "TransactionBuyM5",
    "TransactionSellM5",
    "TransSellM5ChangePercent",
    "TransBuyM5ChangePercent",
    "VolumeM5",
    "VolumeM5ChangePercent",

    # "TransactionBuyH1",
    # "TransactionSellH1",
    # "TransBuyH1ChangePercent",
    # "TransSellH1ChangePercent",
    # "VolumeH1",
    # "VolumeH1ChangePercent",

    # "TransactionBuyH6",
    # "TransactionSellH6",
    # "TransBuyH6ChangePercent",
    # "TransSellH6ChangePercent",
    # "VolumeH6",
    # "VolumeH6ChangePercent",
    
    # "TransactionBuyH24",
    # "TransactionSellH24",
    # "TransBuyH24ChangePercent",
    # "TransSellH24ChangePercent",
    # "VolumeH24",
    # "VolumeH24ChangePercent",

    # "NarrativeScore",
    # "SocialMediaScore",
    # "InfluencerScore",

    ]

    # Outputs = [
    
    #     "TokenPriceM15",
    #     "TokenPriceH1",
    #     "TokenPriceH4",
    #     "TokenPriceH24",

    # ]

    class Outputs:

        PriceChangeM15 = "TokenPriceM15"
        PriceChangeH1 = "TokenPriceH1"
        PriceChangeH4 = "TokenPriceH4"
        PriceChangeH24 = "TokenPriceH24"

Intervals = ['15m', '1h', '6h', '24h']

TimeAndPrice = ["CrawlAt", "TokenPrice"]


class QueryBuilder:

    # PARAMS: contractAddress, WS
    PredictData = '''
        SELECT ci."SmartContractAddress" as "ContractAddress",
        ci."Name" as "TokenName",
        ais."CrawlAt",
        ais."GreedAndFear",
        ais."SMHoldingPercent",
        ais."SMQuantityHold",
        ais."SMRoi",
        ais."SMVolume",
        ais."SMVolumeBuy",
        ais."TokenAntiWhale",
        ais."TokenDumprisk",
        ais."TokenLiquidity",
        ais."TokenLiquiditylock",
        ais."TokenMarketcap",
        ais."TokenPrice",
        ais."TokenTaxbuy",
        ais."TokenTaxsell",
        ais."TotalVolumeBitcoinToday",
        ais."TransactionBuyH1",
        ais."TransactionBuyH24",
        ais."TransactionBuyH6",
        ais."TransactionBuyM5",
        ais."TransactionSellH1",
        ais."TransactionSellH24",
        ais."TransactionSellH6",
        ais."TransactionSellM5",
        ais."TransBuyH1ChangePercent",
        ais."TransBuyH24ChangePercent",
        ais."TransBuyH6ChangePercent",
        ais."TransBuyM5ChangePercent",
        ais."TransSellH1ChangePercent",
        ais."TransSellH24ChangePercent",
        ais."TransSellH6ChangePercent",
        ais."TransSellM5ChangePercent",
        ais."VolumeH1",
        ais."VolumeH1ChangePercent",
        ais."VolumeH24",
        ais."VolumeH24ChangePercent",
        ais."VolumeH6",
        ais."VolumeH6ChangePercent",
        ais."VolumeM5",
        ais."VolumeM5ChangePercent",
        ais."Top10HolderHolding",
        --ads."TokenPriceM15",
        --ads."TokenPriceM30",
        --ads."TokenPriceH1",
        --ads."TokenPriceH4",
        --ads."TokenPriceH24",
        ais."TargetTP50M15",
        ais."TargetTP100M15",
        ais."TargetTP150M15",
        ais."TargetTP200M15",
        ais."TargetTP50H1",
        ais."TargetTP100H1",
        ais."TargetTP150H1",
        ais."TargetTP200H1",
        ais."TargetTP50H4",
        ais."TargetTP100H4",
        ais."TargetTP150H4",
        ais."TargetTP200H4",
        ais."TargetTP50H24",
        ais."TargetTP100H24",
        ais."TargetTP150H24",
        ais."TargetTP200H24",
        ais."Deployed",
        ais."TwitterAvailable",
        ais."BitcoinTrend",
        hg."UnigemScore",
        hg."SocialMediaScore",
        hg."InfluencerScore",
        hg."UnigemScores"->0->'Score' as "NarrativeScore" 

        FROM "AIDataSummary" "ais" 
        INNER JOIN "ContractInfos" "ci" ON ci."Id" = ais."ContractInfoId" 
        LEFT JOIN "HiddenGems" "hg" ON "ci"."Id" = "hg"."ContractInfoId" 
        LEFT JOIN "AIDetailSummary" "ads" ON ci."Id" = ads."ContractInfoId" and ais."CrawlAt" = ads."CrawlAt" 

        WHERE ais."Top10HolderHolding" <= 90
        AND ci."SmartContractAddress" = :contractAddress

        ORDER BY ci."Name" ASC,
        "ais"."CrawlAt" DESC
        LIMIT :WS
        '''
    
    # PARAMS: fromDate, toDate
    TrainData = '''
        SELECT ci."SmartContractAddress" as "ContractAddress",
        ci."Name" as "TokenName",
        ais."CrawlAt",
        ais."GreedAndFear",
        ais."SMHoldingPercent",
        ais."SMQuantityHold",
        ais."SMRoi",
        ais."SMVolume",
        ais."SMVolumeBuy",
        ais."TokenAntiWhale",
        ais."TokenDumprisk",
        ais."TokenLiquidity",
        ais."TokenLiquiditylock",
        ais."TokenMarketcap",
        ais."TokenPrice",
        ais."TokenTaxbuy",
        ais."TokenTaxsell",
        ais."TotalVolumeBitcoinToday",
        ais."TransactionBuyH1",
        ais."TransactionBuyH24",
        ais."TransactionBuyH6",
        ais."TransactionBuyM5",
        ais."TransactionSellH1",
        ais."TransactionSellH24",
        ais."TransactionSellH6",
        ais."TransactionSellM5",
        ais."TransBuyH1ChangePercent",
        ais."TransBuyH24ChangePercent",
        ais."TransBuyH6ChangePercent",
        ais."TransBuyM5ChangePercent",
        ais."TransSellH1ChangePercent",
        ais."TransSellH24ChangePercent",
        ais."TransSellH6ChangePercent",
        ais."TransSellM5ChangePercent",
        ais."VolumeH1",
        ais."VolumeH1ChangePercent",
        ais."VolumeH24",
        ais."VolumeH24ChangePercent",
        ais."VolumeH6",
        ais."VolumeH6ChangePercent",
        ais."VolumeM5",
        ais."VolumeM5ChangePercent",
        ais."Top10HolderHolding",
        ads."TokenPriceM15",
        ads."TokenPriceM30",
        ads."TokenPriceH1",
        ads."TokenPriceH4",
        ads."TokenPriceH24",
        ais."TargetTP50M15",
        ais."TargetTP100M15",
        ais."TargetTP150M15",
        ais."TargetTP200M15",
        ais."TargetTP50H1",
        ais."TargetTP100H1",
        ais."TargetTP150H1",
        ais."TargetTP200H1",
        ais."TargetTP50H4",
        ais."TargetTP100H4",
        ais."TargetTP150H4",
        ais."TargetTP200H4",
        ais."TargetTP50H24",
        ais."TargetTP100H24",
        ais."TargetTP150H24",
        ais."TargetTP200H24",
        ais."Deployed",
        ais."TwitterAvailable",
        ais."BitcoinTrend",
        hg."UnigemScore",
        hg."SocialMediaScore",
        hg."InfluencerScore",
        hg."UnigemScores"->0->'Score' as "NarrativeScore" 
        FROM "AIDataSummary" "ais" 
        INNER JOIN "ContractInfos" "ci" ON ci."Id" = ais."ContractInfoId" 
        LEFT JOIN "HiddenGems" "hg" ON "ci"."Id" = "hg"."ContractInfoId" 
        LEFT JOIN "AIDetailSummary" "ads" ON ci."Id" = ads."ContractInfoId" and ais."CrawlAt" = ads."CrawlAt" 
        WHERE ais."Top10HolderHolding" <= 90 AND ais."CrawlAt" >= :fromDate AND ais."CrawlAt" <= :toDate
        ORDER BY ci."Name" ASC,
        "ais"."CrawlAt" DESC
        '''