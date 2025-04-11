import pulp
from math import gcd
from functools import reduce

def find_proportions(buckets):
    current_gcd = reduce(lambda x, y: gcd(x, y), buckets)
    for d in range(current_gcd, 0, -1):
        if all(b % d == 0 for b in buckets):
            proportions = [b // d for b in buckets]
            if max(proportions) <= 15:
                return d, proportions
    return None, [None] * len(buckets)

def optimize_stockpiles(stockpiles, target_mass, target_grade, tolerance=0.15, bucket_capacity=10):
    """
    Оптимизирует выбор штабелей для смеси руды с выводом пропорций ковшей.
    
    Параметры:
    - stockpiles: список словарей с ключами 'mass' (масса) и 'grade' (содержание золота).
    - target_mass: целевая масса смеси (должна быть кратна `bucket_capacity`).
    - target_grade: целевое содержание золота.
    - tolerance: допустимое отклонение содержания золота.
    - bucket_capacity: вместимость ковша в тоннах (по умолчанию 10).
    
    Возвращает:
    - Словарь {номер_штабеля: количество_ковшей},
    - Итоговая масса,
    - Итоговое содержание золота,
    - Строка пропорции (например, "1:2 x 30") или None, если пропорция не найдена.
    """
    # if target_mass % bucket_capacity != 0:
    #     return None
    
    target_buckets = target_mass // bucket_capacity

    model = pulp.LpProblem("Stockpile_Selection", pulp.LpMinimize)
    indices = range(len(stockpiles))
    
    # Целочисленные переменные: количество ковшей из каждого штабеля
    n = {
        i: pulp.LpVariable(f'n_{i}', lowBound=0, upBound=stockpiles[i].ore_mass_t // bucket_capacity, cat=pulp.LpInteger)
        for i in indices
    }
    y = {i: pulp.LpVariable(f'y_{i}', cat=pulp.LpBinary) for i in indices}
    
    # Цель: минимизировать количество штабелей
    model += pulp.lpSum(y[i] for i in indices)
    
    # Ограничения
    model += pulp.lpSum(n[i] for i in indices) == target_buckets  # Целевое количество ковшей
    
    # Ограничение на содержание золота
    total_gold = pulp.lpSum(n[i] * bucket_capacity * stockpiles[i].gold_concentration_gt for i in indices)
    lower_gold = (target_grade - tolerance) * target_mass
    upper_gold = (target_grade + tolerance) * target_mass
    model += total_gold >= lower_gold
    model += total_gold <= upper_gold
    
    # Связь между n[i] и y[i]
    for i in indices:
        model += n[i] <= y[i] * (stockpiles[i].ore_mass_t // bucket_capacity)
        model += n[i] >= y[i]  # Если y[i] = 1, то n[i] >= 1
    
    # Решение
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if status != pulp.LpStatusOptimal:
        return None
    
    # Извлечение решения
    solution = {i: int(n[i].value()) for i in indices if n[i].value() > 0}
    if not solution:
        return None
    
    # Расчет итоговых массы и содержания золота
    total_mass = sum(buckets * bucket_capacity for buckets in solution.values())
    total_gold_value = sum(
        buckets * bucket_capacity * stockpiles[i].gold_concentration_gt
        for i, buckets in solution.items()
    )
    final_grade = total_gold_value / total_mass
    
    # Поиск пропорций
    buckets = list(solution.values())
    
    def find_proportions(buckets):
        current_gcd = reduce(lambda x, y: gcd(x, y), buckets)
        for d in range(current_gcd, 0, -1):
            if all(b % d == 0 for b in buckets):
                proportions = [b // d for b in buckets]
                if max(proportions) <= 15:
                    return d, proportions
        return None, [None] * len(buckets)
    
    print('h4')
    d, proportions = find_proportions(buckets)
    # if d is None:
    #     return solution, total_mass, final_grade, None
    
    # Формирование строки пропорции
    proportions_dict = {}
    for i, elements in enumerate(solution.items()):
        index, buckets_amount = elements
        proportions_dict[index] = proportions[i]

    proportion_str = ":".join(map(str, proportions)) + f" x {d}"
    print('h5', proportions, solution)

    solution_out = {
        'total_mass': total_mass,
        'concentration': final_grade,
        'used_piles': [
            {
                'index': index+1,
                'buckets_total': buckets_amount,
                'amount': buckets_amount * bucket_capacity, 
                'bucket_proportion': proportions_dict[index]
            }
            for index, buckets_amount in solution.items()
            # for i in range(len(sorted_piles)) if best_scoops[i] > 0
        ],
        # 'score': best_score,
        'used_piles_count': len(solution),
        'cycles_count': d
    }
    print('h6')
    return solution_out
    # return solution, total_mass, round(final_grade, 3), proportion_str


def main():
    # Пример использования
    stockpiles = [
        {'mass': 400, 'grade': 1.5},
        {'mass': 700, 'grade': 2.1},
    ]
    target_mass = 600
    target_grade = 1.8
    tolerance = 0.15
    
    solution, total_mass, final_grade, proportion = optimize_stockpiles(
        stockpiles, target_mass, target_grade, tolerance, bucket_capacity=10
    )
    
    if solution:
        print("Решение найдено:")
        print(f"Ковши из штабелей: {solution}")
        print(f"Итоговая масса: {total_mass} тонн")
        print(f"Содержание золота: {final_grade} г/т")
        print(f"Пропорция: {proportion}")
    else:
        print("Решение не найдено")


if __name__ == '__main__':
       main()
