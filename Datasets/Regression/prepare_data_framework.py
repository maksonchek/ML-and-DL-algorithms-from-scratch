import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
 
class Data_Preparer():
    """
    Класс, создержащий методы, упрощающие работу с данными.
    На вход принимает данный в формате pd.DataFrame

    Методы:
    1) print_nans
    2) get_columns_type
    3) print_feture_types
    4) 
    """
    def __init__(self,data: pd.DataFrame, cat_threshold:int = 10):
        self.data = data
        self.cat_threshold = cat_threshold
        self.num_cols, self.cat_cols, self.other_cols = self.get_columns_type(self.cat_threshold, do_print=False)
        sns.set_style('darkgrid')
        sns.set(font_scale=1.15)

    def print_nans(self) -> list:
        """
        Выводит информацию о признаках с пропусками ф формате: Имя признака, Количество NANов, Относительное кол-во NANов.
        Возвращает список имеён признаков с пропущенными значениями.
        """
        nans = self.data.isnull().sum()
        d_len = len(self.data)
        ret = []
        for n in nans.keys():
            cnans = nans[n]
            if cnans > 0:
                ret.append(n)
                print(f"{n}: Количество NANов: {cnans}, Относительное кол-во NANов: {round(cnans/d_len, 3)}, Тип признака: {self.data[n].dtype}")
        return ret
    
    def get_columns_type(self, cat_threshold:int = 10, do_print = True)->tuple[list, list, list]:
        """
        ! Перед применением метода привести типы колонок к их истинным значениям (числа к числам, строки к объектам)

        cat_threshold - порог, ниже или равно которому признак определяется как категориальный. По умолчанию равен 10
        do_print - печатать или не печатать данные. По умолчанию True

        Метод возвращает списки признаков, разбитые на 3 группы: численные, категориальные, не подходящие ни к одной категории. 
        Категоря признака определяется его типом данных и количеством его значений. Если оно <= cat_threshold и тип - object, то он категориальный. 
        Если > cat_threshold и тип int, то численный. Иначе признак считается неопределенным.

        Порядок возвращения: num_cols,  cat_cols, not_num_not_cat_cols
        """
        num_cols = []
        cat_cols = []
        not_num_not_cat_cols = []
        for col in self.data.columns:
            if do_print:
                print(f"{col}: {self.data[col].dtype}")
                print()
            vc = self.data[col].value_counts()
            if do_print:
                print(vc)
                print()
                print(len(vc))
                print()

            if len(vc) <= cat_threshold:
                cat_cols.append(col)
            elif len(vc) > cat_threshold and self.data[col].dtype != 'object':
                num_cols.append(col)
            else:
                not_num_not_cat_cols.append(col)
        if do_print:
            print(f"Численных признаков: {len(num_cols)}, Категориальных признаков: {len(cat_cols)}, Неопределенных признаков: {len(not_num_not_cat_cols)}")

        return (num_cols,  cat_cols, not_num_not_cat_cols)
    
    def print_feature_types(self) -> list:
        """
        Метод выводит тип признаков и информацию, соответствует ли тип своему истинному значению.
        Провера ведётся путем попытки преобразования признака типа obj в int. Если срабатывает - 
        признак считается некорректным.

        Возвращает список имен признаков, несоовтетсвующих истинному типу.
        """
        f_t = []
        for col in self.data.columns:
            ct = self.data[col].dtype
            is_tr = True
            if ct not in ['int64', 'float64', 'int32', 'float32']:
                try:
                    check = int(self.data[col].sample())
                    is_tr = False
                except:
                    pass
            if is_tr == False:
                f_t.append(col)
            print(f"{col}: Тип признака: {ct}, Совпадает с истинным?: {is_tr}")
        print(f"Признаки, несоответствующие типу: {f_t}")
        return f_t
 

    def drow_cat_cols_pie(self):
        """
        Отрисовывает для каждого категориального признака круговую диаграмму
        """
        for col in self.cat_cols:
            counts = self.data[col].value_counts()
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(col) 
            plt.show()
    
    def drow_cat_cols_hist(self):
        """
        Отрисовывает для каждого категориального признака гистограмму
        """
        for col in self.cat_cols:
            plt.figure()
            self.data[col].value_counts().plot(kind='bar')
            plt.title(col)
            plt.xlabel('Значение')
            plt.ylabel('Частота')
            plt.show()
    
    def drow_num_cols_corr(self):
        plt.figure(figsize=(8,8))
        sns.heatmap(
            self.data[self.num_cols].corr(),
            cmap='RdBu_r', # задаёт цветовую схему
            annot=True, # рисует значения внутри ячеек
            vmin=-1, vmax=1); # указывает начало цветовых кодов от -1 до 1.
    
    def drow_num_cols_hist(self):
        for col in self.num_cols:
            plt.figure() 
            self.data[col].plot(kind='hist', color='green')
            plt.title(col)
            plt.xlabel('Значение')
            plt.ylabel('Частота')
            plt.show()
        