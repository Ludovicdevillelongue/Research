
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import pandas as pd
import json
import time



class CategoryGetter:

    def __init__(self,categories_to_query:list,api:str):
        """
           Parameters
           ----------
                categories_to_query: list -> category selected
                api : str -> api key
         """
        self.categories_to_query=categories_to_query
        self.api=api

    """
    Session Creation
    """
    def headers(self)-> Session:
        """
            Returns
            -------
                session connection
        """
        CMC_PRO_API_KEY = self.api

        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': CMC_PRO_API_KEY,
        }
        cmc_session = Session()
        cmc_session.headers.update(headers)
        return cmc_session

    """
    URL Connection
    """
    def call_coinmarketcap_url(self,url:str, params=None)-> json:
        """
            Parameters
            ----------
                url : str -> URL of coinmarketcap api

            Returns
            -------
                response of API request
        """
        cmc_session=self.headers()
        try:
            response = cmc_session.get(url, params=params)
            return json.loads(response.text)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)

    """
    Dataframe of categories
    """
    def call_api(self):

        res = self.call_coinmarketcap_url('https://pro-api.coinmarketcap.com/v1/cryptocurrency/categories')

        #categories name
        categories_name = [d['name'] for d in res['data']]
        sorted(categories_name)
        categories = res['data']
        categories_dict = {d['name']: d for d in categories}
        categories_dict = {category: categories_dict[category] for category in self.categories_to_query}

        #crypto data for categories
        for name, d in categories_dict.items():
            if 'response' not in d or d['response']['status']['error_code'] != 0:
                params = {}
                params['id'] = d['id']
                params['limit'] = 1000
                d['response'] = self.call_coinmarketcap_url('https://pro-api.coinmarketcap.com/v1/cryptocurrency/category', params)
                time.sleep(0)

        with open("data/coinmarketcap_categories.json", "w") as outfile:
            json.dump(categories_dict, outfile)


    def download_categories(self)-> pd.DataFrame:
        """
            Returns
            -------
                dataframe of cryptos with categories as columns
        """

        # Opening JSON file
        f = open("data/coinmarketcap_categories.json")

        # returns JSON object as
        # a dictionary
        categories_dict = json.load(f)

        class Coin(object):
            def __init__(self, dict_coin):
                self.categories = set()
                for key in dict_coin:
                    setattr(self, key, dict_coin[key])

        coins_dict = {}
        for name, dict_category in categories_dict.items():
            for coin_dict in dict_category['response']['data']['coins']:
                if coin_dict['slug'] not in coins_dict:
                    coin = Coin(coin_dict)
                    coins_dict[coin_dict['slug']] = coin
                coins_dict[coin_dict['slug']].categories.add(name)

        final_coin_dict = {k: v for k, v in coins_dict.items() if (v.cmc_rank and v.cmc_rank <= 300) or
                           any(cat in self.categories_to_query for cat in v.categories)}

        final_cat = set()
        for coin_dict in final_coin_dict.values():
            final_cat = final_cat.union(coin_dict.categories)

        columns = ['id', 'symbol', 'name', 'slug', 'num_market_pairs', 'date_added', 'is_active', 'cmc_rank']
        coin_list = []
        for coin in final_coin_dict.values():
            c_list = []
            for c in columns:
                # print(c)
                c_list.append(getattr(coin, c))
            for cat in final_cat:
                if cat in coin.categories:
                    c_list.append(1)
                else:
                    c_list.append(0)
            coin_list.append(c_list)

        df = pd.DataFrame(coin_list, columns=columns + list(final_cat))
        df.sort_values('cmc_rank', inplace=True)
        df.dropna(inplace=True)
        df = df[df.cmc_rank < 1000]
        df = df.reset_index(drop=True)
        return df



