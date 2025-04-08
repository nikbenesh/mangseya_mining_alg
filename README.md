# mangseya_mining_alg
# Алгоритм подбора штабелей для золотодобывающей компании 

## Описание алгоритма
Принимает на вход целевую концентрацию и массу золота в смеси, а также информацию по штабелям - для каждого: масса руды и концентрация золота, а также вместимость ковша (в тоннах). 
Подбирает штабели, из которых нужно взять руду и выводит их вместе с кол-вом руды, которое нужно взять из каждого. 

## Как запустить 
Нкжно прогнать следующие команды в терминале (находясь в той же папке, что и код): 

`pip install requirments.txt`

`streamlit run sl_interface.py` - для запуска интерфейса 

`python alg.py` - для запуска без интерфейса (ввод и вывод в командной строке) 
