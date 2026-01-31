"""
Точка входа приложения предсказания зарплат (обёртка над regression.app).

Интерфейс: python app.py chain_pattern/x_data.npy
Или: python -m regression.app chain_pattern/x_data.npy
Вывод: список зарплат в рублях (float), по одному на строку.
"""

from regression.app import main

if __name__ == "__main__":
    main()
