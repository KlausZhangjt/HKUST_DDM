# coding: utf-8

import pandas as pd
import tkinter as tk
from tkinter import ttk
import time
import merge_sort
import heap
import AVL_tree


class project2():
    def __init__(self, path):
        self.window = tk.Tk()

        self.data = pd.read_csv(path, header=None)
        self.data.columns = ['VehicleType', 'DetectionTime_O', 'GantryID_O', 'DetectionTime_D', 'GantryID_D',
                             'TripLength', 'TripEnd', 'TripInformation']
        self.data = self.data.dropna()

        self.input1 = tk.Entry(self.window, width=10)
        self.input2 = tk.Entry(self.window, width=10)
        self.input3 = tk.Entry(self.window, width=10)
        self.input4 = tk.Entry(self.window, width=5)
        self.input4_0 = tk.Entry(self.window, width=5)
        self.text1 = tk.Text(width=9, height=1.4)
        self.text2 = tk.Text(width=9, height=1.4)
        self.text3 = tk.Text(width=9, height=1.4)
        self.text4 = tk.Text(width=9, height=1.4)
        self.text5 = tk.Text(width=9, height=1.4)
        self.text6 = tk.Text(width=9, height=1.4)
        self.sort_result_txt = tk.Text(width=95, height=21)

        self.dic = (
            "00", "01", "02", "03", '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
            '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
            '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60')
        self.from_time_1 = tk.StringVar()
        self.time_5_1 = ttk.Combobox(self.window, textvariable=self.from_time_1, width=3)
        self.time_5_1["values"] = self.dic
        self.from_time_2 = tk.StringVar()
        self.time_5_2 = ttk.Combobox(self.window, textvariable=self.from_time_2, width=3)
        self.time_5_2["values"] = self.dic

        self.to_time_1 = tk.StringVar()
        self.to_5_1 = ttk.Combobox(self.window, textvariable=self.to_time_1, width=3)
        self.to_5_1["values"] = self.dic
        self.to_time_2 = tk.StringVar()
        self.to_5_2 = ttk.Combobox(self.window, textvariable=self.to_time_2, width=3)
        self.to_5_2["values"] = self.dic

        self.from_time_11 = tk.StringVar()
        self.time_6_1 = ttk.Combobox(self.window, textvariable=self.from_time_11, width=3)
        self.time_6_1["values"] = self.dic
        self.from_time_22 = tk.StringVar()
        self.time_6_2 = ttk.Combobox(self.window, textvariable=self.from_time_22, width=3)
        self.time_6_2["values"] = self.dic

        self.to_time_11 = tk.StringVar()
        self.to_6_1 = ttk.Combobox(self.window, textvariable=self.to_time_11, width=3)
        self.to_6_1["values"] = self.dic
        self.to_time_22 = tk.StringVar()
        self.to_6_2 = ttk.Combobox(self.window, textvariable=self.to_time_22, width=3)
        self.to_6_2["values"] = self.dic

        self.sort_value = tk.StringVar()
        self.sort_name = ttk.Combobox(self.window, textvariable=self.sort_value)
        self.sort_name["values"] = ("merge_sort", "heap_sort", "AVL_sort")

        self.size_value = tk.StringVar()
        self.size_name = ttk.Combobox(self.window, textvariable=self.size_value)
        self.size_name["values"] = ("20%", "25%", '33%', '50%', '100%')

        self.column_value = tk.StringVar()
        self.column_name = ttk.Combobox(self.window, textvariable=self.column_value)
        self.column_name["values"] = ('VehicleType', 'DetectionTime_O', 'GantryID_O')

    def search1(self):
        text = int(self.input1.get())
        self.text1.delete('1.0', 'end')

        if text not in self.data['VehicleType'].unique():
            self.text1.insert("insert", 0)
        else:
            search_out1 = dict(self.data['VehicleType'].value_counts())[text]
            self.text1.insert("insert", search_out1)

    def search2(self):
        text = self.input2.get()
        self.text2.delete('1.0', 'end')
        if text not in self.data['GantryID_O'].unique():
            self.text2.insert("insert", 0)
        else:
            search_out2 = dict(self.data['GantryID_O'].value_counts())[text]
            self.text2.insert("insert", search_out2)

    def search3(self):
        text = self.input3.get()
        self.text3.delete('1.0', 'end')
        if text not in self.data['GantryID_D'].unique():
            self.text3.insert("insert", 0)
        else:
            search_out3 = dict(self.data['GantryID_D'].value_counts())[text]
            self.text3.insert("insert", search_out3)

    def search4(self):
        from_0, to_0 = float(self.input4.get()), float(self.input4_0.get())
        self.text4.delete('1.0', 'end')
        data4 = self.data['TripLength']
        search_out4 = self.data.shape[0] - data4[data4 < from_0].count() - data4[data4 > to_0].count()
        self.text4.insert("insert", search_out4)

    def search5(self):
        v1 = self.time_5_1.get()
        v2 = self.time_5_2.get()
        v3 = self.to_5_1.get()
        v4 = self.to_5_2.get()
        self.text5.delete('1.0', 'end')
        from_search5 = '2019-08-30 08:' + str(v1) + ':' + str(v2)
        to_search5 = '2019-08-30 08:' + str(v3) + ':' + str(v4)
        data5 = self.data['DetectionTime_O']
        search_out5 = data5.shape[0] - data5[data5 > to_search5].count() - data5[data5 < from_search5].count()
        if search_out5 < 0:
            search_out5 = 0
        self.text5.insert("insert", search_out5)

    def search6(self):
        v1 = self.time_6_1.get()
        v2 = self.time_6_2.get()
        v3 = self.to_6_1.get()
        v4 = self.to_6_2.get()
        self.text6.delete('1.0', 'end')
        from_search6 = '2019-08-30 08:' + str(v1) + ':' + str(v2)
        to_search6 = '2019-08-30 08:' + str(v3) + ':' + str(v4)
        data6 = self.data['DetectionTime_D']
        search_out6 = data6.shape[0] - data6[data6 > to_search6].count() - data6[data6 < from_search6].count()
        if search_out6 < 0:
            search_out6 = 0
        self.text6.insert("insert", search_out6)

    def sort_choose(self, event):
        variable = self.sort_name.get()
        self.sort_result_txt.insert("insert", 'Chosen Sort Type: ' + variable)
        self.sort_result_txt.insert(tk.INSERT, '\n')
        print('Choose Sort Type:', self.sort_name.get())

    def column_choose(self, event):
        variable = self.column_name.get()
        self.sort_result_txt.insert("insert", 'Choose the column of the Data: ' + variable)
        self.sort_result_txt.insert(tk.INSERT, '\n')
        print('Choose the Size of the Data:', self.column_name.get())

    def size_choose(self, event):
        variable = self.size_name.get()
        self.sort_result_txt.insert("insert", 'Choose the Size of the Data: ' + variable)
        self.sort_result_txt.insert(tk.INSERT, '\n')
        print('Choose the Size of the Data:', self.size_name.get())

    def sort_data(self):
        sort = self.sort_name.get()
        size = self.size_name.get()
        size_dic = {'20%': 0.2, '25%': 0.25, '33%': 1 / 3, '50%': 0.5, '100%': 1}
        size = int(size_dic[size] * self.data.shape[0])

        col = self.column_name.get()
        col_dic = {'VehicleType': 0, 'DetectionTime_O': 1, 'GantryID_O': 2}
        col = col_dic[col]
        dt = [row[col] for row in self.data.values.tolist()][:size]
        dt_copy = dt[:]  # To avoid some unexpected mistake.
        result = []

        start_time = time.time()
        if sort == 'merge_sort':
            result = merge_sort.merge_sort(dt_copy)
        elif sort == 'heap_sort':
            result = heap.heap_sort(dt_copy)
        elif sort == 'AVL_sort':
            result = AVL_tree.AVL_sort(dt_copy)
        self.sort_result_txt.insert("insert", 'First 10 entries of sorted result: ')
        self.sort_result_txt.insert(tk.INSERT, '\n')

        for i in range(10):
            self.sort_result_txt.insert("insert", result[i])
            self.sort_result_txt.insert(tk.INSERT, '\n')
        print('First 10 entries of sorted result: ', result[0:10])
        finish_time = time.time() - start_time

        self.sort_result_txt.insert("insert", 'It costs : ' + str(finish_time) + ' s')
        self.sort_result_txt.insert(tk.INSERT, '\n')
        for i in range(len(result) - 1):
            if result[i] > result[i + 1]:
                print("It doesn't perform well!")
                return
        self.sort_result_txt.insert("insert",
                                    f"The sort algorithm {sort} sorts successfully for "
                                    f"column named {self.column_name.get()} with "
                                    f"the size of {len(result)} in {finish_time:.2f} seconds.")
        self.sort_result_txt.insert(tk.INSERT, '\n')
        print(f"The sort algorithm {sort} sorts successfully "
              f"for column named {self.column_name.get()} "
              f"with the size of {len(result)} in {finish_time:.2f} seconds.")

    def clean_data(self):
        self.sort_result_txt.delete('1.0', 'end')

    def show_description_of_the_data(self):
        self.sort_result_txt.insert("insert", 'The first five rows of the data:')
        self.sort_result_txt.insert(tk.INSERT, '\n')
        for i in range(5):
            self.sort_result_txt.insert("insert", self.data.iloc[i, ])
            self.sort_result_txt.insert(tk.INSERT, '\n')

    def show_labels(self):
        tk.Label(self.window, text="Group 3: Li Jiayi Wang Sizhe Zhang Juntao", justify='center',
                 font=('Times', 15, 'bold italic')).place(x=352, y=30)

        tk.Label(self.window, text="Taiwan Traffic Data Inquiry System", justify='center',
                 font=('Times', 20, 'bold italic')).pack()

        tk.Label(self.window, text='SEARCH', font=('Times', 20, 'bold italic')).place(x=10, y=60)
        tk.Label(self.window, text='Here are some choices for you to know what you need.',
                 font=('Times', 17, 'bold italic')).place(x=10, y=84)

        tk.Label(self.window, text='Search how many vehicles are in every Vehicle Type:',
                 font=('Times', 15)).place(x=10, y=110)
        tk.Label(self.window, text='Text here:', font=('Times', 15)).place(x=10, y=135)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=290, y=135)

        tk.Label(self.window, text='Search how many vehicles are in different GantryID_O:',
                 font=('Times', 15)).place(x=10, y=190)
        tk.Label(self.window, text='Text here:', font=('Times', 15)).place(x=10, y=215)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=290, y=215)

        tk.Label(self.window, text='Search how many vehicles are in different GantryID_D:',
                 font=('Times', 15)).place(x=10, y=270)
        tk.Label(self.window, text='Text here:', font=('Times', 15)).place(x=10, y=295)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=290, y=295)

        tk.Label(self.window, text='Search how many vehicles are in an interval of Trip Length:',
                 font=('Times', 15)).place(x=450, y=110)
        tk.Label(self.window, text='From:', font=('Times', 15)).place(x=450, y=135)
        tk.Label(self.window, text='To:', font=('Times', 15)).place(x=600, y=135)
        tk.Label(self.window, text='Both range from 0 to 420', font=('Times', 15)).place(x=700, y=135)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=450, y=165)

        tk.Label(self.window,
                 text='Numbers of vehicle passing from DetectionTime_O in a certain period of time:',
                 font=('Times', 15)).place(x=450, y=190)
        tk.Label(self.window, text='From 2019-08-30 08:', font=('Times', 15)).place(x=450, y=215)
        tk.Label(self.window, text='To     2019-08-30 08:', font=('Times', 15)).place(x=450, y=240)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=700, y=235)

        tk.Label(self.window,
                 text='Numbers of vehicle passing from DetectionTime_D in a certain period of time:',
                 font=('Times', 15)).place(x=450, y=270)
        tk.Label(self.window, text='From 2019-08-30 08:', font=('Times', 15)).place(x=450, y=295)
        tk.Label(self.window, text='To     2019-08-30 08:', font=('Times', 15)).place(x=450, y=320)
        tk.Label(self.window, text='Result:', font=('Times', 15)).place(x=700, y=315)

        tk.Label(self.window, text='SORT', font=('Times', 15, 'bold italic')).place(x=10, y=360)
        tk.Label(self.window, text='Choose one kind of sort algorithm', font=('Times', 15)).place(x=10, y=385)
        tk.Label(self.window, text='Choose the Size of the Data', font=('Times', 15)).place(x=10, y=440)
        tk.Label(self.window, text='Choose the column of the Data', font=('Times', 15)).place(x=10, y=495)

    def show_buttons(self):
        button1 = tk.Button(self.window, text='Search', command=self.search1)
        button1.pack()
        button1.place(x=200, y=130)

        button2 = tk.Button(self.window, text='Search', command=self.search2)
        button2.pack()
        button2.place(x=200, y=215)

        button3 = tk.Button(self.window, text='Search', command=self.search3)
        button3.pack()
        button3.place(x=200, y=295)

        button4 = tk.Button(self.window, text='Search', command=self.search4)
        button4.pack()
        button4.place(x=620, y=160)

        button5 = tk.Button(self.window, text='Search', command=self.search5)
        button5.pack()
        button5.place(x=700, y=210)

        button6 = tk.Button(self.window, text='Search', command=self.search6)
        button6.pack()
        button6.place(x=700, y=290)

        button_sort = tk.Button(self.window, width=10, height=1, bg="white", text='Sort Now', command=self.sort_data)
        button_sort.place(x=80, y=580)

        button_clean = tk.Button(self.window, width=10, height=1, bg="white", text='Clean up the Board',
                                 command=self.clean_data)
        button_clean.place(x=80, y=620)

        button_show = tk.Button(self.window, width=10, height=1, bg="white", text='Show the Board',
                                command=self.show_description_of_the_data)
        button_show.place(x=80, y=660)

    def show_others(self):
        self.input1.pack()
        self.input1.place(x=100, y=135)
        self.text1.place(x=340, y=136)
        self.input2.pack()
        self.input2.place(x=100, y=215)
        self.text2.place(x=340, y=216)
        self.input3.pack()
        self.input3.place(x=100, y=295)
        self.text3.place(x=340, y=296)
        self.input4.pack()
        self.input4.place(x=500, y=133)
        self.input4_0.pack()
        self.input4_0.place(x=630, y=133)
        self.text4.place(x=500, y=165)

        self.time_5_1.current(0)
        self.time_5_1.pack()
        self.time_5_1.place(x=590, y=215)
        self.time_5_2.current(0)
        self.time_5_2.pack()
        self.time_5_2.place(x=640, y=215)
        self.to_5_1.current(0)
        self.to_5_1.pack()
        self.to_5_1.place(x=590, y=240)
        self.to_5_2.current(0)
        self.to_5_2.pack()
        self.to_5_2.place(x=640, y=240)
        self.text5.place(x=750, y=240)

        self.time_6_1.current(0)
        self.time_6_1.pack()
        self.time_6_1.place(x=590, y=295)
        self.time_6_2.current(0)
        self.time_6_2.pack()
        self.time_6_2.place(x=640, y=295)
        self.to_6_1.current(0)
        self.to_6_1.pack()
        self.to_6_1.place(x=590, y=320)
        self.to_6_2.current(0)
        self.to_6_2.pack()
        self.to_6_2.place(x=640, y=320)
        self.text6.place(x=750, y=320)
        self.sort_result_txt.place(x=300, y=410)

        self.sort_name.current(0)
        self.sort_name.bind("<<ComboboxSelected>>", self.sort_choose)
        self.sort_name.pack()
        self.sort_name.place(x=40, y=410)

        self.size_name.current(0)
        self.size_name.bind("<<ComboboxSelected>>", self.size_choose)
        self.size_name.pack()
        self.size_name.place(x=40, y=465)

        self.column_name.current(0)
        self.column_name.bind("<<ComboboxSelected>>", self.column_choose)
        self.column_name.pack()
        self.column_name.place(x=40, y=520)

    def work(self):
        self.window.title('5051 Project 2 of Group 3')
        self.window.geometry('1000x720')
        self.show_labels()
        self.show_buttons()
        self.show_others()
        self.window.mainloop()


if __name__ == '__main__':
    project2(path="TDCS_M06A_20190830_080000.csv").work()
