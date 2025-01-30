import json
import pandas as pd
import re
from gensim import corpora, models, similarities


def tokenize(name: str) -> list[str]:
    tokens_raw = re.split("\\s|-|\\(|\\)|\\/|\\+", name.lower())
    for i in range(len(tokens_raw)):
        # More often than not offers from vendors write internal storage size separately, "256 GB" instead of "256GB", while the latter is more common for shop strings.
        if re.match("[0-9]+gb", tokens_raw[i]):
            tokens_raw.append(re.match("[0-9]+", tokens_raw[i])[0])
            tokens_raw.append("gb")
    return [token for token in tokens_raw if token]


def apply_results(df, shops_to_be):
    for i in range(4):  # Creating new columns
        df[f"Price {i+1}"] = ""
        df[f"Vendor {i+1}"] = ""
    for ind_df, vendors_data in shops_to_be.items():
        for num, offer in enumerate(vendors_data):
            df.loc[ind_df, f"Price {num+1}"] = offer["price"]
            df.loc[ind_df, f"Vendor {num + 1}"] = offer["vendor"]
    return df


def parse_price(vendor_string):
    res = re.search(r"[0-9]{4,5}", vendor_string)
    return res[0] if res else -1  # -1 is supposed to indicate error, this will be useful for filtering false positives.


threshold = 0.63  # Minimum distance between strings to be considered.

if __name__ == '__main__':
    # Next two files were filled manually to speed up the parsing process and to make it more precise. 
    # This file contains vendors, categories of their products and their respective range of indices within the table.
    with open("vendors_ranges.json") as f:
        ranges_vendor = json.load(f)
    # Same as the previous one, but only categories and ranges are present.
    with open("shops_ranges.json") as f:
        ranges_shop = json.load(f)
    shops_csv = pd.read_csv('shops.csv', sep=',', names=["name", "ext_code", "manufacturer", "model", "RAM", "sim_num",
                                                         "type", "processor", "color", "manufacturer_code", "int_mem"],
                            encoding="utf8", header=0)
    # We will use TF-IDF model to measure the distance between shop strings and vendor strings.
    shops_csv["tokens"] = shops_csv["name"].apply(lambda name: tokenize(name))
    d = corpora.Dictionary(shops_csv["tokens"])
    corpus = [d.doc2bow(text) for text in shops_csv["tokens"]]
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(d.token2id))
    # Now let's create a dictionary, indices of shop items which were found also in vendor file will serve as keys. This
    # will indicate that exactly those indices must be extended by vendor and price information, which we will store as 
    # values of the dictionary.
    shops_to_be = {}
    vendors_csv = pd.read_csv('vendors.csv', sep=',', names=["main", "vendor"], encoding="utf8", header=0)
    for vendor_name, categories in ranges_vendor.items():
        for category_name, category_range in categories.items():
            for i in range(category_range["min"], category_range["max"]):
                text = vendors_csv["main"][i]
                if not isinstance(text, str):  # Sometimes wrong type is assigned if the cell is empty, let's skip them.
                    continue
                price = parse_price(text)
                if price == -1:  # -1 indicates error, as was mentioned earlier.
                    continue
                max_likeness = -1
                probable_name = ""
                probable_name_index = -1
                # It's worthless to search for related strings in different categories, let's set up boundaries.
                for i in range(ranges_shop[category_name]["min"], ranges_shop[category_name]["max"]):
                    new_likeness = index[tfidf[d.doc2bow(tokenize(text))]][i]
                    if new_likeness > max_likeness:
                        max_likeness = new_likeness
                        probable_name = shops_csv["name"][i]
                        probable_name_index = i
                # At the end of this cycle we have found the most probable choice among shop strings. Let's check if the
                # best result is any good.
                if max_likeness < threshold:
                    continue
                if probable_name_index not in shops_to_be:
                    shops_to_be[probable_name_index] = [{"likeness": max_likeness, "vendor": vendor_name, "price": price}]
                else:
                    # Since we cycle through vendor strings, we may run into a case where shop string already has
                    # vendor string assigned to it, but the one we have just encountered is actually more likely to be
                    # the correct one. Though if new vendor string is related to another vendor, then it's totally OK to
                    # add this information on top of existing one. That's why we initialize a flag to watch if the vendor
                    # of current entry is present.
                    vendor_listed = False
                    for offer in shops_to_be[probable_name_index]:
                        if offer["vendor"] == vendor_name:
                            vendor_listed = True
                            if offer["likeness"] < max_likeness:
                                offer["likeness"] = max_likeness
                                offer["price"] = price
                    if not vendor_listed:
                        shops_to_be[probable_name_index].append({"likeness": max_likeness, "vendor": vendor_name, "price": price})
    apply_results(shops_csv, shops_to_be)  # Now let's just throw what we've learned on top of existing shop table and save it for future use.
    shops_csv.pop("tokens")
    shops_csv.to_excel("result.xlsx")
