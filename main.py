import pandas

def read_csv():
    df = pandas.read_csv("Contrastive.csv")

    # group the dataframe by art_style and create a dictionary of dataframes
    art_style_groups = df.groupby('art_style')
    dfs_dict = {}
    for art_style, group in art_style_groups:
        dfs_dict[art_style] = group

    # print the new dataframes
    # for key, value in dfs_dict.items():
    #    print(f"Dataframe for {key}:")
    #    print(value)
    #    print()

    return dfs_dict



read_csv()
