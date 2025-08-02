# ML-store-sales-forecast
# Problem Definition:
მონაცემებში წარმოდგენილია ვოლმარტის გაყიდვები 45 სხვადასხვა მაღაზიაში, რომლებიც სხვადასხვა რეგიონებში მდებარეობს.
თითოეულ მაღაზიაში არის რამდენიმე დეპარტამენტი და დავალება არის დეპარტამენტის გაყიდვების პროგნოზირება თითოეულ მაღაზიაში.

ვოლმარტი მთელი წლის განმავლობაში ატარებს სხვადასხვა ტიპის ფასდაკლებებს. 
ეს ფასდაკლებები წინ უსწრებს მნიშვნელოვან დღესასწაულებს, რომელთა შორის არიან: სუპერ ბოული, შრომის დღე, მადლიერების დღე და შობა.(4 ყველაზე დიდი დღესასწაული უკვეთება).

train.csv - სატრენინგო მონაცემები (მაღაზია, დეპარტამენტი, თარიღი, გაყიდვები)
test.csv - სატესტო მონაცემები პროგნოზისთვის
features.csv - დამატებითი ფაქტორები (ტემპერატურა, საწვავის ფასი, CPI, უმუშევრობა, ფასდაკლებები)
stores.csv - მაღაზიების მეტამონაცემები (ტიპი, ზომა)

# EDA
გვაქვს 3 სხვადასხვა ფაილი. 
train.csv
features.csv
stores.csv
ამიტომ პირველ რიგში ვაკეთებთ ამათ დაჯოინებას.

გვაქვს 3 ტიპის მაღაზია: A,B,C
<img width="1150" height="770" alt="image" src="https://github.com/user-attachments/assets/0a500d47-d44a-4a53-a305-7c88eade20fc" />
ვხედავთ, რომ A ტიპის მაღაზიების მედიანები უფრო მაღალია, ვიდრე სხვა ტიპის მაღაზიების მედიანები, ამიტომ A ტიპის მაღაზიის ყოველკვირეული გაყიდვები სხვა ტიპის მაღაზიებთან შედარებით მეტია.

ვნახეთ, როგორ არის დამოკიდებული weekly_sales დღესასწაულებზე
<img width="1176" height="700" alt="image" src="https://github.com/user-attachments/assets/650d4755-dd27-4f1e-ba7f-aaad49aefe5b" />
გაყიდვები იზრდება დღესასწაულების პერიოდში.

<img width="1268" height="567" alt="image" src="https://github.com/user-attachments/assets/1b94c365-68e5-4752-9854-a462c6672347" />
გვაქვს სეზონურობა

დიკი ფულერის ტესტის გამოყენებით შეგვიძლია შევამოწმოთ სტაციოანლურობა.(ნულოვანი ჰიპოთეზის დაშვება უარყოფა...)

სტაციონალური მონაცემები გვჭირდება ისეთი მოდელისთვის როგორიცაა მაგ: ARIMA

<img width="411" height="134" alt="image" src="https://github.com/user-attachments/assets/31b53bf2-624d-4323-90e3-e22d6f0281ea" />

ADF<0 p<0.05 სტაციონალურია 

კორელაციის მატრიცა ავაგეთ და დავაკვირდით გვქონდა თუ არა კორელირებული მონაცემები.
<img width="869" height="764" alt="image" src="https://github.com/user-attachments/assets/d0a4e1d2-924b-459b-abe1-56b647238446" />

არ აქვთ მკვეთრი კორელაცია

დავაკვირდით რა გავლენა ქონდა CPI, fuel price, unemployment, temperature ტიპის მონაცემებს weekly sale-ზე
<img width="928" height="750" alt="image" src="https://github.com/user-attachments/assets/f83258bd-fdd3-4c5e-ba03-fe2aa92648d1" />

ვნახეთ როგორ იცვლებოდა გაყიდვები დღესასწაულები კვირაში.
<img width="995" height="405" alt="image" src="https://github.com/user-attachments/assets/fe4d5aee-252b-46cb-8d74-dea1e3cd8978" />
<img width="943" height="408" alt="image" src="https://github.com/user-attachments/assets/128a3dbf-8ea4-4809-a91f-e66b2ca1096b" />
<img width="1012" height="405" alt="image" src="https://github.com/user-attachments/assets/d57637c4-26e9-4d00-ae3c-ba2d3e895705" />
<img width="988" height="416" alt="image" src="https://github.com/user-attachments/assets/7b7f1b14-ecc4-4d21-89e2-b95e3094cbff" />

ანუ დღესასწაულის დღეები უმეტესად მთელი კვირის გაყიდვებზე ახდენს გავლენას

დავაკვირდით გამოტოვებულ მნიშვნელობებს:
<img width="996" height="541" alt="image" src="https://github.com/user-attachments/assets/fd0d0bfd-9f63-4259-b420-4ca84c6659c0" />

ასევე ვნახეთ თუ როგორ იცვლებოდა გაყიდვები სეზონებისა და თვეების მიხედვით. 
რომელ სეზონზე ან თვეში იყო ხოლმე ყველაზე მაღალი. 


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

# Classical Statistical Time-Series Models
# Data Preprocessing
Merger კლასი
აერთიანებს სხვადასხვა მონაცემის ფაილებს 
ასუფთავებს და  თარიღების ფორმატს ასწორებს (datetime)
ქმნის ერთიან დატასეტს ანალიზისთვის

DateTimeFeatureExtractor კლასი
ამოიღებს თარიღიდან კომპონენტებს (დღე, თვე, წელი)
გარდაქმნის ტემპერატურას ფარენჰეიტიდან ცელსიუსში
ქმნის დროითი ფაქტორების ცვლადებს

HolidayFeatureGenerator კლასი
ამოიცნობს დღესასწაულების კვირებს:
Super Bowl (თებერვალი)
Labor Day (სექტემბერი)
Thanksgiving (ნოემბერი)
Christmas (დეკემბერი)
ითვლის დღეების რაოდენობას დღესასწაულებამდე
ქმნის სეზონურ ფაქტორებს

NaFiller კლასი
ავსებს გამოტოვებულ მნიშვნელობებს:
ფასდაკლებები → 0
ეკონომიკური ინდიკატორები → საშუალო მნიშვნელობა

CategoryMapper კლასი
მაღაზიის ტიპი: A=3, B=2, C=1
დღესასწაული: True=1, False=0

StoreDataProcessor კლასი
აგრეგირებს მონაცემებს მაღაზიების მიხედვით
ითვლის დეპარტამენტების წილს მთლიან გაყიდვებში

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

# N-BEATS 
N-BEATS (Neural Basis Expansion Analysis for Time Series) არის თანამედროვე ღრმა სწავლების მოდელი დროითი მწკრივების პროგნოზირებისთვის, რომელიც შეიქმნა 2019 წელს.
არის univariate- არ იყენებენ ეგზოგენურ ცვლადებს.
 N-BEATS არქიტექტურა
 Input (12 weeks) → Block 1 → Block 2 → Block 3 → Output (8 weeks forecast)
 N-BEATS მოდელის არქიტექტურა
1. Trend Block (ტრენდი)
დანიშნულება: სწავლობს ზოგად, გრძელვადიან ტრენდს
მეთოდი: პოლინომიური ფიტი (polynomial fitting)
სკოუპი: დიდი დროის მონაკვეთი, general direction

2. Seasonality Block (სეზონურობა)
დანიშნულება: ამოიცნობს სეზონურ პატერნებს და ციკლებს
მეთოდი: ფურიე სერიის დახმარებით (Fourier series)
სკოუპი: მოკლე მონაკვეთის ტენდენციები (მაგ. დღესასწაულების სპაიკები)

3. Identity/Generic Blocks (რეზიდუალები)
დანიშნულება: სწავლობს დარჩენილ patterns-ებს
მეთოდი: ზოგადი neural network ბლოკები
სკოუპი: რაც trend-მა და seasonality-მ ვერ ისწავლა
