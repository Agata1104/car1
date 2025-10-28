import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_prepare_data(filepath, target_column):

    data = pd.read_csv(filepath)

    # Разделение на признаки и целевую переменную
    X = data.select_dtypes(include=[np.number]).drop(columns=[target_column])
    y = data[target_column]

    # Удаление столбцов с пропущенными значениями
    X = X.dropna(axis=1)

    return X, y


def remove_highly_correlated_features(X, threshold=0.9):
    # Удаление сильно коррелированных признаков
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop)


def visualize_pca(X_pca, y):
    # Визуализация PCA компонент и целевой переменной
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis', alpha=0.7)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('SalePrice')
    plt.colorbar(sc, label='SalePrice')
    plt.title('3D график PCA признаков и SalePrice')
    plt.show()


def train_and_evaluate_model(X_train, X_test, y_train, y_test, alphas):
    # Обучение модели и оценка качества
    rmse_list = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return rmse_list


def plot_rmse_vs_alphas(alphas, rmse_list):
    # График зависимости RMSE от alpha
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, rmse_list, marker='o')
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('RMSE')
    plt.title('Зависимость ошибки RMSE от коэффициента регуляризации Lasso')
    plt.grid(True)
    plt.show()


def main():
    DATA_PATH = 'C:/Users/Lenovo/Downloads/AmesHousing.csv'
    TARGET = 'SalePrice'

    # Загрузка и подготовка данных
    X, y = load_and_prepare_data(DATA_PATH, TARGET)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Удаление сильно коррелированных признаков
    X_reduced = remove_highly_correlated_features(X_scaled)

    # Визуализация PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_reduced)
    visualize_pca(X_pca, y)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )

    # Подбор параметра регуляризации
    alphas = np.logspace(-4, 0, 50)
    rmse_list = train_and_evaluate_model(X_train, X_test, y_train, y_test, alphas)
    plot_rmse_vs_alphas(alphas, rmse_list)

    # Выбор лучшей модели
    best_alpha = alphas[np.argmin(rmse_list)]
    print(f"Лучший alpha: {best_alpha}")

    best_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
    best_model.fit(X_train, y_train)

    # Анализ важности признаков
    coef = pd.Series(best_model.coef_, index=X_reduced.columns)
    coef_abs = coef.abs()
    most_influential_feature = coef_abs.idxmax()
    print(f"Признак с наибольшим влиянием на целевое значение: {most_influential_feature}")
    print(f"Коэффициент: {coef[most_influential_feature]}")


if __name__ == "__main__":
    main()



