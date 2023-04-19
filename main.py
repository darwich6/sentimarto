import pandas

def read_csv():
    df = pandas.read_csv("Contrastive.csv");
    print(df.to_csv(index=False))

read_csv()
