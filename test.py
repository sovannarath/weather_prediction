from prettytable import PrettyTable

my_table = PrettyTable()
my_table.field_names = ["No.", "Name", "Grade", "Age"]
my_table.add_row([1, "Bob", 6, 11])
my_table.add_row([2, "Freddy", 4, 10])
my_table.add_row([3, "John", 7, 13])
my_table.sortby = 'Age'
my_table.reversesort = True
print(my_table)