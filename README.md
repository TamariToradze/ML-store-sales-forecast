# ML-store-sales-forecast
Problem Definition:
მონაცემებში წარმოდგენილია ვოლმარტის გაყიდვები 45 სხვადასხვა მაღაზიაში, რომლებიც სხვადასხვა რეგიონებში მდებარეობს.
თითოეულ მაღაზიაში არის რამდენიმე დეპარტამენტი და დავალება არის დეპარტამენტის გაყიდვების პროგნოზირება თითოეულ მაღაზიაში.

ვოლმარტი მთელი წლის განმავლობაში ატარებს სხვადასხვა ტიპის ფასდაკლებებს. 
ეს ფასდაკლებები წინ უსწრებს მნიშვნელოვან დღესასწაულებს, რომელთა შორის არიან: სუპერ ბოული, შრომის დღე, მადლიერების დღე და შობა.(4 ყველაზე დიდი დღესასწაული უკვეთება).

#EDA


# LightGBM
აერთიანებს მონაცემებს 
უმკლავდება დუბლირებულ სვეტებს (მაგ. IsHoliday_x, IsHoliday_y) 
ვავსებთ nan მონაცემებს
მაღაზიის ტიპს (Type) კოდირებისთვის ვიყენებთ LabelEncoder-ის გამოყენებით.

ქმნის უნიკალურ კოდირებულ ID-ს მაღაზია-დეპარტამენტის (Store_Dept) კომბინაციებისთვის.

გამომდინარე იქედან რომ მოდელს არ აქვს დროის აღქმა, ვამატებთ შემდეგ სვეტებს. 
დროითი ცვლადები: დღე, კვირა, თვე, წელი, კვარტალი, წლის რომელი დღეა, თვის რომელი კვირაა(უკეთესად რომ დაიჭიროს ციკლური და სეზონური პატერნები)

დეკემბრის და მეოთხე კვარტალის ცალკე მონიშვნა (IsDecember, Is(Oct – Dec)). რადგან ამ პერიოდში გვაქვს გაყიდვების პიკები.

გვაქვს ჩამორჩენის მახასიათებლები (lag_1, lag_2 და სხვ.) წარსული გაყიდვების ტენდენციების ასახვისთვის.

ვთლით ცვლილებას ((this_week - prev_week) / prev_week),  მეტ ინფორმაციას აძლევს ვიდრე უბრალოდ წინა კვირის გაყიდვები. ხედავს მოდელი იზრდება თუ მცირდება ტენდენცია.

გაყიდვების შეფარდება საშუალოსთან (sales_vs_mean)
რამდენად განსხვავდება მიმდინარე გაყიდვა გრძელვადიანი საშუალოდან.

მთლიანად დაწეული ფასები (total_markdown)
რამდენი ტიპის markdown იყო გამოყენებული (markdown_count)


ეკონომიკური წნეხი:
აერთიანებს ინფლაციის ინდექსსა (CPI) და უმუშევრობას (Unemployment)

ტემპერატურის ეფექტი:
ცივ/ცხელ დღეებზე მახასიათებლები, ტემპერატურის კვადრატი

დასვენების დღეების ეფექტი:
ინტერაქცია დასვენების დღეებსა და დეპარტამენტებს/მაღაზიის ტიპს შორის

მაღაზიის ზომის გავლენა:
მაღაზიის ზომის ინტერაქცია საწვავის ფასსა და მაღაზიის ტიპთან

ვალაგებთ მონაცემებს თარიღის მიხედვით
def temporal_cross_validation_split(train_data, n_splits=5):
    -დროის მიხედვით split-ები, არა რანდომული
    -ყოველი fold-ი იყენებს მომავალ მონაცემებს validation-ისთვის
შემდეგ Optuna-ს გამოყენებით ვარჩევთ საუკეთესო ჰიპერპარამეტრებს
საბოლოო შედეგები:
MAE: 1294.69
RMSE: 2402.67
R²: 0.9888
WMAE: 1340.13

MLflow tracking:
LightGBM_Feature_Engineering:
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/7baaad47ced147e695af5a1be0f1aee2

LightGBM_Cleaning:
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/449257015f1d43e4b3809efe8ea60006

LightGBM_CrossValidation:
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/e91be9d39afa4b1ab3d283628ad6e1a9

LightGBM_Final_Training
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/97342c4589924bbca5d94ee0fe696c4c

LightGBM_Prediction
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/409b0ab8cd134be693f103b1729e819d

LightGBM_Model_Selection
https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/17/runs/d8cbe7007ab74468a794ba730226321f

# ARIMA
ARIMA = AutoRegressive Integrated Moving Average
ARIMA(p,d,q) სადაც:
- p = AR (AutoRegressive) - წარსული მნიშვნელობების რაოდენობა
- d = I  (Integrated) - დიფერენციაციის ხარისხი (სტაციონარობისთვის)  
- q = MA (Moving Average) - წარსული შეცდომების რაოდენობა
ტრენინგის მონაცემები:
ARIMA მოდელი მხოლოდ Weekly_Sales ცვლადზე არის დატრენინგებული

თითოეული მაღაზიისთვის ყველა დეპარტამენტის Weekly_Sales ერთმანეთს ემატება კონკრეტული თარიღისთვის.
45 მაღაზია გვაქვს dataset-ში
თითოეული მაღაზიისთვის ყველა დეპარტამენტის Weekly_Sales ჯამდება ერთ თარიღზე.
შედეგად მიიღება: თითოეული მაღაზიისთვის თარიღების მიხედვით ჯამური Weekly_Sales
დატრენინგებულია 45 სხვადასხვა მოდელი.
საუკეთესო კონფიგურაციად ირჩევს იმას, რომელსაც აქვს ყველაზე დაბალი საშუალო MAPE ყველა მაღაზიაზე.

# Prophet
y(t) = g(t) + s(t) + h(t)
სადაც:
g(t) - ტრენდის კომპონენტი, რომელიც ამსახავს სამიზნე ცვლადის არაწრფივ ქცევას
s(t) - სეზონურობის კომპონენტი, რომელიც იჭერს პერიოდულ შაბლონებს
h(t) - დღესასწაულების კომპონენტი, რომელიც მოდელირებს სპეციალური მოვლენების ეფექტებს

Prophet Input → 
ds (Date)
y (Weekly_Sales) 
IsHoliday
SuperbowlWeek
LaborDayWeek
ThanksgivingWeek
ChristmasWeek
grid search-ის გამოყენებით შერჩეულია საუკეთესო პარამეტრები 
<img width="1057" height="781" alt="image" src="https://github.com/user-attachments/assets/8ad18398-7cb8-47e4-a038-55e8ff009201" />

