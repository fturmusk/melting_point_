from src.preprocess import read_file
from src.models train_model

def main():
    path1 = r"C:\Users\batoc\Desktop\Github\melting_point_\data\train.csv"
    path2 = r"C:\Users\batoc\Desktop\Github\melting_point_\data\test.csv"

    df_test, df_train, id_, y_train = read_file(path1,path2)
    train_model(df_train, y_train, df_test, id_)

if __name__ == "__main__":
    main()


