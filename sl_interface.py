import streamlit as st
import pandas as pd
import numpy as np
from alg import Pile, calculate_mixture, find_optimal_combinations

st.set_page_config(page_title="Алгоритм подбора смеси руды", layout="wide")

st.title("Алгоритм подбора оптимальной смеси руды для Мангазея майнинг")

# Боковая панель для ввода параметров
with st.sidebar:
    st.header("Параметры")
    
    # Ввод размера ковша
    scoop_size_t = st.number_input("Размер ковша (т)", min_value=0.1, value=5.0, step=0.1)
    
    # Ввод целевых параметров
    target_mass = st.number_input("Целевая масса руды (т)", min_value=0.1, value=100.0, step=1.0)
    target_concentration = st.number_input("Целевая концентрация золота (г/т)", min_value=0.01, value=1.0, step=0.01)
    
    # Ввод допустимого отклонения
    allowed_deviation = st.number_input("Допустимое отклонение концентрации (г/т)", min_value=0.01, value=0.15, step=0.01)
    
    # Выбор способа ввода данных о штабелях
    input_method = st.radio("Способ ввода данных о штабелях", ["Вручную", "Из CSV файла"])

# Основная область для ввода данных и отображения результатов
if input_method == "Вручную":
    st.header("Ввод данных о штабелях")
    
    # Ввод количества штабелей
    piles_number = st.number_input("Количество штабелей", min_value=1, value=3, step=1)
    
    # Создание полей для ввода данных о каждом штабеле
    piles = []
    for i in range(int(piles_number)):
        st.subheader(f"Штабель {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ore_mass_t = st.number_input(f"Масса руды (т)", min_value=0.1, value=50.0, step=1.0, key=f"mass_{i}")
        
        with col2:
            gold_concentration_gt = st.number_input(f"Концентрация золота (г/т)", min_value=0.01, value=1.0, step=0.01, key=f"conc_{i}")
        
        with col3:
            preference = st.slider(f"Предпочтительность (%)", min_value=0, max_value=100, value=100, step=1, key=f"pref_{i}") / 100
        
        piles.append(Pile(ore_mass_t, gold_concentration_gt, index=i+1, preference=preference))
    
    # Кнопка для запуска алгоритма
    if st.button("Найти оптимальные комбинации"):
        solutions = find_optimal_combinations(piles, target_mass, target_concentration, scoop_size_t, allowed_deviation)
        
        if not solutions:
            st.error("Не удалось найти подходящие комбинации штабелей.")
        else:
            st.success(f"Найдено {len(solutions)} вариантов комбинаций штабелей.")
            
            for i, solution in enumerate(solutions):
                with st.expander(f"Вариант {i+1}"):
                    st.write(f"Итоговая масса руды: {solution['total_mass']:.2f} т")
                    st.write(f"Итоговая концентрация золота: {solution['concentration']:.2f} г/т")
                    st.write(f"Отклонение от целевой концентрации: {abs(solution['concentration'] - target_concentration):.2f} г/т")
                    st.write(f"Использовано штабелей: {solution['used_piles_count']}")
                    
                    # Создаем DataFrame для отображения используемых штабелей
                    used_piles_data = []
                    for pile in solution['used_piles']:
                        used_piles_data.append({
                            "Штабель": pile['index'],
                            "Ковшей": pile['scoops'],
                            "Масса (т)": f"{pile['amount']:.2f}"
                        })
                    
                    st.table(pd.DataFrame(used_piles_data))
                    
                    # Визуализация результатов
                    st.subheader("Визуализация")
                    
                    # График концентрации
                    st.write("Концентрация золота в смеси")
                    concentration_data = pd.DataFrame({
                        "Параметр": ["Целевая", "Полученная"],
                        "Концентрация (г/т)": [target_concentration, solution['concentration']]
                    })
                    st.bar_chart(concentration_data.set_index("Параметр"))
                    
                    # График массы
                    st.write("Масса руды")
                    mass_data = pd.DataFrame({
                        "Параметр": ["Целевая", "Полученная"],
                        "Масса (т)": [target_mass, solution['total_mass']]
                    })
                    st.bar_chart(mass_data.set_index("Параметр"))
                    
                    # График распределения ковшей по штабелям
                    st.write("Распределение ковшей по штабелям")
                    scoops_data = pd.DataFrame({
                        "Штабель": [pile['index'] for pile in solution['used_piles']],
                        "Ковшей": [pile['scoops'] for pile in solution['used_piles']]
                    })
                    st.bar_chart(scoops_data.set_index("Штабель"))

else:  # Ввод из CSV файла
    st.header("Загрузка данных из CSV файла")
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Предварительный просмотр данных:")
            st.dataframe(df.head())
            
            # Проверка наличия необходимых столбцов
            required_columns = ['ore_mass_t', 'gold_concentration_gt', 'index']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"В файле отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
                st.info("Файл должен содержать следующие столбцы: ore_mass_t, gold_concentration_gt, index")
            else:
                # Создание объектов Pile из данных CSV
                piles = []
                for _, row in df.iterrows():
                    preference = row.get('preference', 1.0)  # Если столбец preference отсутствует, используем 1.0
                    piles.append(Pile(row['ore_mass_t'], row['gold_concentration_gt'], index=row['index'], preference=preference))
                
                st.success(f"Загружено {len(piles)} штабелей")
                
                # Кнопка для запуска алгоритма
                if st.button("Найти оптимальные комбинации"):
                    solutions = find_optimal_combinations(piles, target_mass, target_concentration, scoop_size_t, allowed_deviation)
                    
                    if not solutions:
                        st.error("Не удалось найти подходящие комбинации штабелей.")
                    else:
                        st.success(f"Найдено {len(solutions)} вариантов комбинаций штабелей.")
                        
                        for i, solution in enumerate(solutions):
                            with st.expander(f"Вариант {i+1}"):
                                st.write(f"Итоговая масса руды: {solution['total_mass']:.2f} т")
                                st.write(f"Итоговая концентрация золота: {solution['concentration']:.2f} г/т")
                                st.write(f"Отклонение от целевой концентрации: {abs(solution['concentration'] - target_concentration):.2f} г/т")
                                st.write(f"Использовано штабелей: {solution['used_piles_count']}")
                                
                                # Создаем DataFrame для отображения используемых штабелей
                                used_piles_data = []
                                for pile in solution['used_piles']:
                                    used_piles_data.append({
                                        "Штабель": pile['index'],
                                        "Ковшей": pile['scoops'],
                                        "Масса (т)": f"{pile['amount']:.2f}"
                                    })
                                
                                st.table(pd.DataFrame(used_piles_data))
                                
                                # Визуализация результатов
                                st.subheader("Визуализация")
                                
                                # График концентрации
                                st.write("Концентрация золота в смеси")
                                concentration_data = pd.DataFrame({
                                    "Параметр": ["Целевая", "Полученная"],
                                    "Концентрация (г/т)": [target_concentration, solution['concentration']]
                                })
                                st.bar_chart(concentration_data.set_index("Параметр"))
                                
                                # График массы
                                st.write("Масса руды")
                                mass_data = pd.DataFrame({
                                    "Параметр": ["Целевая", "Полученная"],
                                    "Масса (т)": [target_mass, solution['total_mass']]
                                })
                                st.bar_chart(mass_data.set_index("Параметр"))
                                
                                # График распределения ковшей по штабелям
                                st.write("Распределение ковшей по штабелям")
                                scoops_data = pd.DataFrame({
                                    "Штабель": [pile['index'] for pile in solution['used_piles']],
                                    "Ковшей": [pile['scoops'] for pile in solution['used_piles']]
                                })
                                st.bar_chart(scoops_data.set_index("Штабель"))
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {str(e)}")
    
    else:
        st.info("Загрузите CSV файл с данными о штабелях")
        st.markdown("""
        CSV файл должен содержать следующие столбцы:
        - `ore_mass_t` - масса руды в тоннах
        - `gold_concentration_gt` - концентрация золота в г/т
        - `index` - индекс штабеля
        - `preference` (опционально) - предпочтительность штабеля от 0 до 1
        """)

# Информация о формате CSV файла
with st.expander("Информация о формате CSV файла"):
    st.markdown("""
    CSV файл должен содержать следующие столбцы:
    - `ore_mass_t` - масса руды в тоннах
    - `gold_concentration_gt` - концентрация золота в г/т
    - `index` - индекс штабеля
    - `preference` (опционально) - предпочтительность штабеля от 0 до 1
    
    Пример:
    ```
    ore_mass_t,gold_concentration_gt,index,preference
    50.0,1.2,1,1.0
    30.0,0.8,2,0.8
    70.0,1.5,3,0.6
    ```
    """)
    
    # Пример CSV файла
    example_df = pd.DataFrame({
        'ore_mass_t': [50.0, 30.0, 70.0],
        'gold_concentration_gt': [1.2, 0.8, 1.5],
        'index': [1, 2, 3],
        'preference': [1.0, 0.8, 0.6]
    })
    st.dataframe(example_df)
    
    # Кнопка для скачивания примера
    csv = example_df.to_csv(index=False)
    st.download_button(
        label="Скачать пример CSV файла",
        data=csv,
        file_name="example_piles.csv",
        mime="text/csv"
    )
