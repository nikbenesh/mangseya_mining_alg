import pandas as pd
import numpy as np
from itertools import combinations
import math
from lp_alg import find_proportions


class Pile:
    def __init__(self, ore_mass_t, gold_concentration_gt, index=0, preference=1.0):
        self.ore_mass_t = ore_mass_t
        self.gold_concentration_gt = gold_concentration_gt
        self.index = index
        self.gold_mass_g = self.ore_mass_t * gold_concentration_gt
        self.preference = preference
    
    def __str__(self):
        return f'Index: {self.index}, Ore mass (t): {self.ore_mass_t}, Gold concentration (g/t): {self.gold_concentration_gt}, Preference: {self.preference}'


def read_csv_piles(filename='piles.csv'):
    df = pd.read_csv(filename)
    piles = []
    for index, row in df.iterrows():
        pile = Pile(row['ore_mass_t'], row['gold_concentration_gt'], index=row['index'])
        piles.append(pile)
    return piles


def input_piles():
    piles_number = int(input("Введите количество штабелей: "))
    piles = []
    for i in range(piles_number):
        pile_mass_t = float(input(f"Введите массу руды в штабеле {i+1} (т): "))
        pile_concentration_gt = float(input(f"Введите концентрацию золота в штабеле {i+1} (г/т): "))
        preference = float(input(f"Введите предпочтительность штабеля {i+1} (0-100%): ")) / 100
        gold_mass_g = pile_mass_t * pile_concentration_gt
        print(f"Масса штабеля {i+1}: {pile_mass_t} т. Масса золота (г): {gold_mass_g}")
        piles.append(Pile(pile_mass_t, pile_concentration_gt, index=i+1, preference=preference))
    return piles


def calculate_mixture(piles, used_scoops, scoop_size_t, target_concentration):
    """Рассчитывает параметры смеси при заданном количестве ковшей из каждого штабеля"""
    total_ore_mass = 0
    total_gold_mass = 0
    
    for i in range(len(piles)):
        # Рассчитываем массу руды из текущего штабеля
        ore_from_pile = min(used_scoops[i] * scoop_size_t, piles[i].ore_mass_t)
        total_ore_mass += ore_from_pile
        
        # Рассчитываем массу золота из текущего штабеля
        gold_concentration = piles[i].gold_mass_g / piles[i].ore_mass_t
        total_gold_mass += ore_from_pile * gold_concentration
    
    mixture_concentration = total_gold_mass / total_ore_mass if total_ore_mass > 0 else 0
    
    return total_ore_mass, mixture_concentration


def find_optimal_combinations(piles, target_mass, target_concentration, scoop_size_t, allowed_concentration_deviation=0.15, allowed_mass_deviation=0.05, max_solutions=3):
    """Находит оптимальные комбинации штабелей"""
    # Сортировка штабелей по предпочтительности
    sorted_piles = sorted(piles, key=lambda p: -p.preference)
    
    # Список для хранения решений
    solutions = []
    
    # Функция оценки качества решения
    def solution_score(total_mass, concentration, used_scoops):
        # Отклонение концентрации
        concentration_error = abs(concentration - target_concentration)
        
        # Отклонение массы
        mass_error = abs(total_mass - target_mass) / target_mass if target_mass > 0 else float('inf')
        
        # Штраф за отклонение от целевых параметров
        if concentration_error > allowed_concentration_deviation:
            return float('-inf')
        
        # if mass_error > allowed_mass_deviation:
        #     return float('-inf')
        
        # Предпочтительность используемых штабелей
        preference_score = sum(sorted_piles[i].preference * (used_scoops[i] > 0) for i in range(len(sorted_piles)))
        
        # Штраф за использование большого количества штабелей
        used_piles_count = sum(1 for s in used_scoops if s > 0)
        pile_count_penalty = used_piles_count / len(sorted_piles) if len(sorted_piles) > 0 else 0
        
        # Итоговая оценка: больше = лучше
        # Приоритет набора массы: если масса близка к целевой, увеличиваем оценку
        mass_bonus = 10.0 if mass_error < 0.01 else 0.0
        
        # Приоритет концентрации: если концентрация близка к целевой, увеличиваем оценку
        concentration_bonus = 5.0 if concentration_error < 0.15 else 0.0
        
        return mass_bonus + concentration_bonus + preference_score - pile_count_penalty
    
    # Рассчитываем максимальное количество ковшей для целевой массы
    target_scoops = math.ceil(target_mass / scoop_size_t)
    
    # Проверяем комбинации с разным количеством штабелей
    for k in range(1, len(sorted_piles) + 1):  # Убрали ограничение на 5 штабелей
        for pile_indices in combinations(range(len(sorted_piles)), k):
            # Задача оптимизации: найти оптимальное количество ковшей для выбранных штабелей
            
            best_scoops = None
            best_score = float('-inf')
            best_mass = 0
            best_concentration = 0
            
            # Максимальное количество ковшей из каждого штабеля
            max_scoops_per_pile = [math.floor(sorted_piles[i].ore_mass_t / scoop_size_t) if i in pile_indices else 0 
                                  for i in range(len(sorted_piles))]
            
            # Грубый перебор для простоты
            current_scoops = [0] * len(sorted_piles)
            
            # Добавляем ковши один за другим, выбирая наиболее подходящий штабель
            max_iterations = target_scoops * 3  # Увеличили множитель для более широкого поиска
            for _ in range(max_iterations):
                best_step_score = float('-inf')
                best_step_idx = -1
                
                # Проверяем, какой штабель лучше всего добавить следующим
                for idx in pile_indices:
                    if current_scoops[idx] < max_scoops_per_pile[idx]:
                        # Пробуем добавить один ковш из этого штабеля
                        current_scoops[idx] += 1
                        mass, concentration = calculate_mixture(sorted_piles, current_scoops, scoop_size_t, target_concentration)
                        
                        # Проверяем, не превышает ли масса целевую слишком сильно
                        mass_error = abs(mass - target_mass) / target_mass if target_mass > 0 else float('inf')
                        if mass_error <= allowed_mass_deviation:
                            score = solution_score(mass, concentration, current_scoops)
                            
                            # Возвращаем как было
                            current_scoops[idx] -= 1
                            
                            if score > best_step_score:
                                best_step_score = score
                                best_step_idx = idx
                        else:
                            # Возвращаем как было
                            current_scoops[idx] -= 1
                
                # Если нашли улучшение, применяем его
                if best_step_idx >= 0:
                    current_scoops[best_step_idx] += 1
                    mass, concentration = calculate_mixture(sorted_piles, current_scoops, scoop_size_t, target_concentration)
                    
                    # Если решение лучше предыдущего, запоминаем его
                    score = solution_score(mass, concentration, current_scoops)
                    if score > best_score:
                        best_score = score
                        best_scoops = current_scoops.copy()
                        best_mass = mass
                        best_concentration = concentration
                    
                    # Если достигли или превысили целевую массу и концентрация в допуске,
                    # то смотрим, можно ли убрать лишние ковши
                    if mass >= target_mass and abs(concentration - target_concentration) <= allowed_concentration_deviation:
                        # Пробуем убирать ковши по одному, начиная с наименее предпочтительных штабелей
                        for remove_idx in sorted(pile_indices, key=lambda i: sorted_piles[i].preference):
                            if current_scoops[remove_idx] > 0:
                                current_scoops[remove_idx] -= 1
                                new_mass, new_concentration = calculate_mixture(sorted_piles, current_scoops, scoop_size_t, target_concentration)
                                
                                # Если всё ещё в допуске и масса достаточная, сохраняем изменение
                                if new_mass >= target_mass and abs(new_concentration - target_concentration) <= allowed_concentration_deviation:
                                    mass, concentration = new_mass, new_concentration
                                    score = solution_score(mass, concentration, current_scoops)
                                    if score > best_score:
                                        best_score = score
                                        best_scoops = current_scoops.copy()
                                        best_mass = mass
                                        best_concentration = concentration
                                else:
                                    # Возвращаем ковш обратно
                                    current_scoops[remove_idx] += 1
                else:
                    # Если нет улучшений, заканчиваем
                    break
            
            # Если нашли допустимое решение, добавляем его в список
            if best_score > float('-inf'):
                d, proportions = find_proportions(best_scoops)
                print(best_scoops)
                print(proportions)
                solution = {
                    'total_mass': best_mass,
                    'concentration': best_concentration,
                    'used_piles': [
                        {
                            'index': sorted_piles[i].index,
                            'buckets_total': best_scoops[i],
                            'amount': best_scoops[i] * scoop_size_t,
                            'bucket_proportion': proportions[i]

                        }
                        for i in range(len(sorted_piles)) if best_scoops[i] > 0
                    ],
                    'score': best_score,
                    'cycles_count': d,
                    'used_piles_count': sum(1 for s in best_scoops if s > 0)
                }
                solutions.append(solution)
    
    # Сортируем решения по оценке и возвращаем лучшие
    solutions.sort(key=lambda s: (-s['score'], s['used_piles_count']))
    return solutions[:max_solutions]


def main():
    print("Алгоритм подбора оптимальной смеси руды для золотодобывающей компании")
    
    # Ввод размера ковша
    scoop_size_t = float(input("Введите размер ковша (т): "))
    
    # Ввод целевых параметров
    target_mass = float(input("Введите целевую массу руды (т): "))
    target_concentration = float(input("Введите целевую концентрацию золота (г/т): "))
    
    # Спрашиваем, хочет ли пользователь изменить допустимое отклонение концентрации
    change_concentration_deviation = input("Хотите изменить допустимое отклонение концентрации? (да/нет): ").lower()
    if change_concentration_deviation == 'да':
        allowed_concentration_deviation = float(input("Введите допустимое отклонение концентрации (г/т): "))
    else:
        allowed_concentration_deviation = 0.15
    
    # Спрашиваем, хочет ли пользователь изменить допустимое отклонение массы
    change_mass_deviation = input("Хотите изменить допустимое отклонение массы? (да/нет): ").lower()
    if change_mass_deviation == 'да':
        allowed_mass_deviation = float(input("Введите допустимое отклонение массы (доля от целевой массы): "))
    else:
        allowed_mass_deviation = 0.05
    
    # Ввод количества комбинаций
    max_solutions = int(input("Введите желаемое количество комбинаций (по умолчанию 3): ") or "3")
    
    # Выбор способа ввода данных о штабелях
    input_method = input("Выберите способ ввода данных о штабелях (1 - из CSV, 2 - вручную): ")
    
    if input_method == '1':
        filename = input("Введите имя CSV файла (по умолчанию 'piles.csv'): ") or 'piles.csv'
        piles = read_csv_piles(filename)
    else:
        piles = input_piles()
    
    # Поиск оптимальных комбинаций
    solutions = find_optimal_combinations(piles, target_mass, target_concentration, scoop_size_t, 
                                        allowed_concentration_deviation, allowed_mass_deviation, max_solutions)
    
    # Вывод результатов
    if not solutions:
        print("Не удалось найти подходящие комбинации штабелей.")
    else:
        print(f"\nНайдено {len(solutions)} комбинаций штабелей:")
        for i, solution in enumerate(solutions):
            print(f"\nВариант {i+1}:")
            print(f"Итоговая масса руды: {solution['total_mass']:.2f} т")
            print(f"Отклонение от целевой массы: {abs(solution['total_mass'] - target_mass):.2f} т ({abs(solution['total_mass'] - target_mass) / target_mass * 100:.1f}%)")
            print(f"Итоговая концентрация золота: {solution['concentration']:.2f} г/т")
            print(f"Отклонение от целевой концентрации: {abs(solution['concentration'] - target_concentration):.2f} г/т")
            print(f"Использовано штабелей: {solution['used_piles_count']}")
            print("Используемые штабели:")
            for pile in solution['used_piles']:
                print(f"Штабель {pile['index']} - {pile['scoops']} ковшей ({pile['amount']:.2f} т)")


if __name__ == "__main__":
    main()


