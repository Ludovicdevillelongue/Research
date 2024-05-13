
from datetime import date
import pandas as pd
from Autoencoder_Asset_Pricing.data_management.request_categories import CategoryGetter
import time
from datetime import timedelta, datetime
from collections import defaultdict



def get_category(universe_select: str,coinmarketcap_api_key:str)-> list:
    """
        Parameters
        ----------
            universe_select : str -> Category to retrieve like NFT
            coinmarketcap_api_key : str -> API key to use CoinMarketCap

        Returns
        -------
            list of crypto related to the category selected
    """

    categories_to_query = [universe_select]
    #use of request_categories
    category_getter = CategoryGetter(categories_to_query, coinmarketcap_api_key)
    category_getter.call_api()
    df_coin_cat = category_getter.download_categories()
    #categories begin at column 8
    df_cat = df_coin_cat.iloc[:, 8:]
    df_cat.index = df_coin_cat["symbol"]
    #if there is a 1 a crypto belongs to a category
    df_uni = df_cat.loc[df_cat[universe_select] == 1]
    #list of crypto for the selected category
    uni_list = list(df_uni.index)
    return uni_list
