# نبدأ بإنشاء كلاس ديال سيارة
class Car:
    # خاصية الانشاء (Constructor)
    def __init__(self, brand, model, year):
        self.brand = brand  # ماركة السيارة
        self.model = model  # موديل السيارة
        self.year = year    # سنة الصنع

    # ميثود باش نعرضو المعلومات
    def display_info(self):
        return f"Car: {self.brand} {self.model}, Year: {self.year}"

# دابا نقدروا نصاوبوا اوبجي جديد ونخدمو عليه
my_car = Car("Toyota", "Corolla", 2020)
print(my_car.display_info())  # Car: Toyota Corolla, Year: 2020
