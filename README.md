# ML-store-sales-forecast
# Problem Definition:
მონაცემებში წარმოდგენილია ვოლმარტის გაყიდვები 45 სხვადასხვა მაღაზიაში, რომლებიც სხვადასხვა რეგიონებში მდებარეობს.
თითოეულ მაღაზიაში არის რამდენიმე დეპარტამენტი და დავალება არის დეპარტამენტის გაყიდვების პროგნოზირება თითოეულ მაღაზიაში.

ვოლმარტი მთელი წლის განმავლობაში ატარებს სხვადასხვა ტიპის ფასდაკლებებს. 
ეს ფასდაკლებები წინ უსწრებს მნიშვნელოვან დღესასწაულებს, რომელთა შორის არიან: სუპერ ბოული, შრომის დღე, მადლიერების დღე და შობა.(4 ყველაზე დიდი დღესასწაული უკვეთება).

- train.csv - სატრენინგო მონაცემები (მაღაზია, დეპარტამენტი, თარიღი, გაყიდვები)
- test.csv - სატესტო მონაცემები პროგნოზისთვის
- features.csv - დამატებითი ფაქტორები (ტემპერატურა, საწვავის ფასი, CPI, უმუშევრობა, ფასდაკლებები)
- stores.csv - მაღაზიების მეტამონაცემები (ტიპი, ზომა)

ჩვენი კვლევის ძირითადი მიზანი იყო სხვადასხვა მოდელის ერთმანეთისთვის შედარება და მათი ანალიზი. ამიტომაც, შევარჩიეთ შემდეგი მოდელები:
- LightGBM
- NBeats
- Prophet
- ARIMA
- SARIMAX
- DLinear
- XGBoost
- Linear Regression
- Random Forest
  
ჩვენი მიდგომა მოიცავდა შემდეგ ნაბიჯებს:
1. მონაცემთა წინასწარი დამუშავება: მონაცემთა გაწმენდა, NaN-ების ჩანაცვლება/შევსება და კატეგორიული ცვლადების გარდაქმნა.
2. feature-ების ინჟინერია: ახალი feature-ების შექმნა, რომლებიც ასახავს დროის ტენდენციებს, სეზონურობას და სხვა ფაქტორებს.
3. მოდელების ტრენინგი: მოდელების დატრენინგება train data-ზე და მათი მეტრიკების შეფასება.
4. ჰიპერპარამეტრების ოპტიმიზაცია: მოდელების მუშაობის გაუმჯობესება ჰიპერპარამეტრების ოპტიმიზაციის სხვადასხვა მეთოდის გამოყენებით.
5. MLflow: ექსპერიმენტების დალოგვა
6. საბოლოო მოდელის შერჩევა: საუკეთესო მოდელის არჩევა მათივე მეტრიკების მიხედვით.

# რეპოზიტორიის სტრუქტურა
1. EDA.ipynb
2. model_experiment_LightGBM.ipynb
3. model_experiment_NBEATS.ipynb
4. model_experiment_Priphet.ipynb
5. model_experiment_arima.ipynb
6. model_experiment_DLinear.ipynb
7. model_experiment_SARIMA.ipynb
8. model_experiment_XGBoost.ipynb
9. model_inference_xgboost.ipynb
10. README.md

.ipynb ფაილები შეიცავს Notebook-ებს თითოეული მოდელისთვის, სადაც დეტალურადაა აღწერილი მონაცემთა დამუშავება, feature engineering/selection, მოდელის ტრენინგი და მათი მეტრიკები.
readme.md - პროექტის მიმოხილვა


# Feature Engineering
## კატეგორიული ცვლადების რიცხვითში გადაყვანა
- მაღაზიის ტიპი (Type): გარდავქმენით კატეგორიულ ცვლადად (category) და გამოვიყენეთ One-Hot Encoding (A, B, C ტიპებისთვის),ეს საშუალებას აძლევს მოდელსს, გაითვალისწინოს მაღაზიის ტიპის გავლენა მის გაყიდვებზე.
- დღესასწაულის მაჩვენებელი (IsHoliday): გარდავქმენით ბინარულ ცვლადად (0/1), სადაც 1 მიუთითებს დღესასწაულზე და 0 ჩვეულებრივ დღეზე... გაერთიანების შედეგად წარმოქმნილი ორი IsHoliday სვეტი (IsHoliday_x, IsHoliday_y) გავაერთიანეთ ერთ სვეტად, ამოვიღეთ ზედმეტი სვეტები და გარდავქმენით int ტიპად.

## NAN მნიშვნელობების დამუშავება
- MarkDown სვეტები: MarkDown1-დან MarkDown5-მდე სვეტებში NaN მნიშვნელობები შევავსეთ 0-ით, რადგან ეს სვეტები წარმოადგენს ფასდაკლებებს, რომლებიც, თუ NaN-ია, ჩავთვალეთ 0-ად.
- CPI და უმუშევრობა (Unemployment): NaN მნიშვნელობები შევავსეთ თითოეული სვეტის მედიანური მნიშვნელობით, რათა თავიდან აგვეცილებინა გადახრები მონაცემებში.

## Cleaning მიდგომები 
- ამოვიღეთ ჩანაწერები, სადაც Weekly_Sales უარყოფითი იყო (შესაძლოა რამე შეცდომაა ან თანხის დაბრუნებას ნიშნავს, რაც არ გამოგვადგებოდა დატრენინგებაში)
- Handling outliers: გამოვიყენეთ IQR (Interquartile Range) მეთოდი გაყიდვების outlier-ების 'დასაჭერად'. 
- outlier-ები, რომლებიც არ იყო დღესასწაულის/ნოემბერ-დეკემბრის პერიოდში, შევცვალეთ ზედა ზღვრით, რათა თავიდან აგვეცილებინა მოდელის overfitting.

# Feature Selection
- train_full და test_full დავალაგეთ Store, Dept და Date-ის მიხედვით, რათა სწორად გამოვთვალოთ ლაგ-ფიჩერები.
- გამოვიყენეთ კორელაციის ანალიზი, რათა გამოვავლინოთ ყველაზე მნიშვნელოვანი ფიჩერები.

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
<img width="1068" height="658" alt="image" src="https://github.com/user-attachments/assets/15a291a8-aa1a-44c1-8c91-463155486dec" />

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
რომელ სეზონზე ან თვეში იყო ხოლმე ყველაზე მაღალი. (ოქტომბერი-დეკემბერი იმატებს ხოლმე)
დღესასწაულის დღეებს და აქციებს აშკარა გავლენა ქონდათ გაყიდვებზე.
გაყიდვები განსხვავდება გაღაზიებისა და დეპარტამენტების მიხედვით, მაგალითად პროდუქტებში გაყიდვები მაღალია ხოლმე.
Markdown-ები აშკარა გავლენას ახდენენ გაყიდვებზე. განსაკუთრებით კი markdown1 და markdown4.

# Training
## ტესტირებული მოდელები:
ჩვენ გამოვცადეთ ცხრა განსხვავებული მოდელი:
1. Linear Regression: საბაზისო მოდელი, რომელიც კარგად იჭერს წრფივ დამოკიდებულებებს. (ამ პრობლემისთვის არც თუ ისე შესაფერისი...)
2. Random Forest: მოდელი, რომელიც ეფექტურია არაწრფივი დამოკიდებულების დასაფიქსირებლად.
3. XGBoost: წინა ორ მოდელთან შედარებით გამოირჩევა მაღალი სიზუსტით
4. LightGBM: ოპტიმიზებული მოდელი დიდი მონაცემებისთვის
5. SARIMA: time series მოდელი, რომელიც ითვალისწინებს სეზონურობას
6. ARIMA: time series მოდელი არასეზონური ტენდენციებისთვის
7. NBeats: ნეირონული ქსელის მოდელი, სპეციალურად შექმნილი დროის მწკრივის პროგნოზირებისთვის
8. DLinear: წრფივი ნეირონული მოდელი დროის მწკრივებისთვის
9. Prophet: Facebook-ის მიერ შექმნილი მოდელი, რომელიც ავტომატურად ამუშავებს სეზონურობას და დღესასწაულებს

## Linear Regression
Linear Regression გამოვიყენე საბაზისო მოდელად, რომელიც ცდილობს დაიჭიროს წრფივი დამოკიდებულება Weekly_Sales-სა და სხვადასხვა featureb-ს შორის (მაგ., Store_Sales_Mean, HolidaySeason, CPI), ამის გამო ამ მოდელის დატრენინგება არ იყო კარგი იდეა ამ პრობლემისთვის... 

შედეგები: 
- Linear Regression (Train) Metrics:
  - MAE: 7160.7770
  - MSE: 136859787.0574
  - RMSE: 11698.7088
  - R2: 0.6523
  - MAPE: 12529.0283
  - WMAE: 7160.7770
- Linear Regression (Val) Metrics:
  - MAE: 6735.7998
  - MSE: 89510679.6375
  - RMSE: 9461.0084
  - R2: 0.7218
  - MAPE: 36485.8000
  - WMAE: 6735.7998
 
https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/1/runs/f1a17c382ebe417897e4752d783284a6

https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/1?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

<img width="786" height="483" alt="image" src="https://github.com/user-attachments/assets/123fb3ed-7944-4172-826a-90ebe3c26ea8" />

## Random Forest
Random Forest აერთიანებს 100 გადაწყვეტილების ხეს არაწრფივი ურთიერთობების დასაფიქსირებლად. გამოვიყენე პარამეტრები: n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42 და n_jobs=-1 .
პარამეტრების საუკეთესო კომბინაცია იყო n_estimators=100, max_depth=15 და min_samples_split=5, რამაც გააუმჯობესა მოდელის სიზუსტე და შეამცირა overfitting

შედეგები: 
- Random Forest (Train) Metrics:
  - MAE: 1230.6036
  - MSE: 6340282.2819
  - RMSE: 2517.9917
  - R2: 0.9839
  - MAPE: 1079.2664
  - WMAE: 1230.6036
- Random Forest (Val) Metrics:
  - MAE: 1661.9067
  - MSE: 9792586.2384
  - RMSE: 3129.3108
  - R2: 0.9696
  - MAPE: 3713.2180
  - WMAE: 1661.9067
 
https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/2/runs/d929068ee28b462b844e078c7789c66d

https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/2?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

<img width="789" height="492" alt="image" src="https://github.com/user-attachments/assets/f0145e85-d281-4c55-bc3b-c5ccc59ee002" />


## XGBoost
მოდელი დავატრენინგე შემდეგი ჰიპერპარამეტრებით:
xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'eval_metric': 'mae'
    }
მოდელი ტრენინგდა იმავე 80%-იან dataზე,overfitting-ის თავიდან ასაცილებლად

შედეგები: 
- XGBoost (Train) Metrics:
  - MAE: 1859.0164
  - MSE: 9748211.0595
  - RMSE: 3122.2125
  - R2: 0.9752
  - MAPE: 2419.1423
  - WMAE: 1859.0164
- XGBoost (Val) Metrics:
  - MAE: 2290.7544
  - MSE: 14200514.7450
  - RMSE: 3768.3570
  - R2: 0.9559
  - MAPE: 7993.1636
  - WMAE: 2290.7544
 
  https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/3/runs/fd3eda7176954ba881ba2f2f41b2babf

  https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/3

  <img width="787" height="491" alt="image" src="https://github.com/user-attachments/assets/d4811982-b433-4f73-bd82-6ff79e10e93e" />


მოდელების შესადარებლად გამოვიყენოთ შემდეგი გრაფიკები:

<img width="1188" height="592" alt="image" src="https://github.com/user-attachments/assets/a0988d95-5a8b-4eed-98a4-b3d6d65c0598" />

<img width="1395" height="595" alt="image" src="https://github.com/user-attachments/assets/e2499faa-fa58-4c97-bdba-1253fd6da327" />

<img width="1062" height="73" alt="image" src="https://github.com/user-attachments/assets/58dd2c67-7e64-4ddc-ba4d-b9ab6e63cc86" />

## LightGBM
აერთიანებს მონაცემებს 
უმკლავდება დუბლირებულ სვეტებს (მაგ. IsHoliday_x, IsHoliday_y) 
ვავსებთ nan მონაცემებს
მაღაზიის ტიპს (Type) კოდირებისთვის ვიყენებთ LabelEncoder-ის გამოყენებით.

ქმნის უნიკალურ კოდირებულ ID-ს მაღაზია-დეპარტამენტის (Store_Dept) კომბინაციებისთვის.

გამომდინარე იქედან რომ მოდელს არ აქვს დროის აღქმა, ვამატებთ შემდეგ სვეტებს. 
დროითი ცვლადები: დღე, კვირა, თვე, წელი, კვარტალი, წლის რომელი დღეა, თვის რომელი კვირაა(უკეთესად რომ დაიჭიროს ციკლური და სეზონური პატერნები)

ქმნის ძირითადი დღესასწაულებს: Super Bowl, Labor Day, Thanksgiving, Christmas.

გვაქვს ბინარული ფლაგი, რომელიც აღნიშნავს, არის თუ არა კვირა სადღესასწაულო, რადგან გაყიდვები მკვეთრად იცვლება დღესასწაულების პერიოდში.

დეკემბრის და მეოთხე კვარტალის ცალკე მონიშვნა (IsDecember, Is(Oct – Dec)). რადგან ამ პერიოდში გვაქვს გაყიდვების პიკები.

გვაქვს ჩამორჩენის მახასიათებლები (lag_1, lag_2 და სხვ.) წარსული გაყიდვების ტენდენციების ასახვისთვის, ავტოკორელაციის. 
მაგალითად გაყიდვების ზრდა თუ გვქონდა წარსულში, მოსალოდნელიია, რომ ახლაც გაიზრდება.

Rolling Window Features (მოძრავი ფანჯრის სტატისტიკები): მოძრავი საშუალო, სტანდარტული გადახრა,
მინიმუმი და მაქსიმუმი Weekly_Sales-ისთვის რათა გამოვავლინოთ მოკლევადიანი და საშუალოვადიანი ტრენდები.
მაგალითად 4 კვირის საშუალო უფრო მდგრადია ვიდრე 1 კვირის მონაცემი, იკვეთება რამდენად ქაოსურია გაყიდვები.

ვთლით ცვლილებას ((this_week - prev_week) / prev_week),  მეტ ინფორმაციას აძლევს ვიდრე უბრალოდ წინა კვირის გაყიდვები. ხედავს მოდელი იზრდება თუ მცირდება ტენდენცია.

გაყიდვების შეფარდება საშუალოსთან (sales_vs_mean)
რამდენად განსხვავდება მიმდინარე გაყიდვა გრძელვადიანი საშუალოდან.

მთლიანად დაწეული ფასები (total_markdown)
რამდენი ტიპის markdown იყო გამოყენებული (markdown_count)

ციკლურობიდან გამომდინარე cos, sin გამოყენებაც შეიძლებოდა.

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
- MAE: 1294.69
- RMSE: 2402.67
- R²: 0.9888
- WMAE: 1340.13

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

<img width="1013" height="71" alt="image" src="https://github.com/user-attachments/assets/e61aa087-20d9-4475-87b9-af14973f59c1" />


# Classical Statistical Time-Series Models
## Data Preprocessing
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

## ARIMA
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

მთავარი მომენტები:
 თითოეული Store-ისთვის:

Store-ის ჯამური კვირეული გაყიდვები (Weekly_Sales) → Time Series
123 კვირა training, დანარჩენი validation
ARIMA მოდელი ფიტდება მხოლოდ ამ store-ის გაყიდვების ისტორიაზე

ARIMA რას აკვირდება:

p: წინა p კვირის გაყიდვებს (ავტორეგრესია)
d: ტრენდის ცვლილებებს (differencing)
q: წინა q შეცდომების გავლენას

მაგალითად ARIMA(2,1,1):
Sales[t] = c + φ₁*(Sales[t-1] - Sales[t-2]) + φ₂*(Sales[t-2] - Sales[t-3]) + θ₁*error[t-1] + error[t]

 Parameter Selection:

20 კომბინაცია იტესტება: (0,0,1), (0,1,0), (1,1,1), (2,1,1) და ა.შ.
ყოველი კომბინაციისთვის ითვლება ყველა store-ზე საშუალო MAPE
ირჩევა ის კომბინაცია, რომელსაც აქვს ყველაზე დაბალი საშუალო შეცდომა

Department-ების მიღება:

Store-ის პროგნოზი * Department-ის ისტორიული წილი
მაგ.: Store პროგნოზი = $100,000, Department 1-ის წილი = 15% → Department პროგნოზი = $15,000

ანუ მოდელი არ იყენებს external features-ს (ტემპერატურა, CPI და ა.შ.), მხოლოდ sales-ის time series პატერნებს აკვირდება!

Grid Search - 20 განსხვავებული ARIMA პარამეტრის კომბინაციის ტესტირება
მაღაზია-სპეციფიკური მოდელები - თითოეული მაღაზიისთვის ცალკე ARIMA მოდელი
ვალიდაცია - 123 კვირა ტრენინგისთვის, დანარჩენი ვალიდაციისთვის


 ARIMA - "ერთი სეზონურობა"

იძულებულია ავირჩიოს:
- კვირეული სეზონურობა (შაბათი > ორშაბათი)
- ან წლიური სეზონურობა (დეკემბერი > იანვარი)
- მაგრამ ვერ ივნებს ორივეს ერთდროულად!
- 
  • Best Mean MAPE: 6.0226
  
<img width="1059" height="78" alt="image" src="https://github.com/user-attachments/assets/c75f5a58-742b-4a30-aff2-6920c1d446c4" />

## Prophet
y(t) = g(t) + s(t) + h(t)
სადაც:

g(t) - ტრენდის კომპონენტი, რომელიც ამსახავს სამიზნე ცვლადის არაწრფივ ქცევას

s(t) - სეზონურობის კომპონენტი, რომელიც იჭერს პერიოდულ შაბლონებს

h(t) - დღესასწაულების კომპონენტი, რომელიც მოდელირებს სპეციალური მოვლენების ეფექტებს

Prophet Input → 
- ds (Date)
- y (Weekly_Sales) 
- IsHoliday
- superbowlWeek
- LaborDayWeek
- ThanksgivingWeek
- ChristmasWeek

grid search-ის გამოყენებით შერჩეულია საუკეთესო პარამეტრები 

Store-Specific Models: ყოველი store-ისთვის ცალკე Prophet მოდელი
მიზეზი: ყოველი store-ს უნიკალური patterns და seasonality აქვს

Store-Level Prediction: Prophet იძლევა store-level ტოტალურ პრედიქციას
Department Ratios: ისტორიული department-wise განაწილება
Final Prediction: store_prediction * department_ratio

 View experiment at: https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/14

 Prophet - "მრავალი სეზონურობა"
 
ერთდროულად ხედავს:
- კვირეული: შაბათი > ორშაბათი
- და წლიური: დეკემბერი > იანვარი
- და თვიური: თვის დასაწყისი > თვის ბოლო
- და დღიური: საღამო > დილა

  <img width="1054" height="78" alt="image" src="https://github.com/user-attachments/assets/93bd3db3-6f21-4518-9815-28d16cd89f6c" />


## SARIMAX
გამოვიყენეთ SARIMAX მოდელი Walmart-ის გაყიდვების (Weekly Sales) პროგნოზირებისთვის, როგორც time series ანალიზის მეთოდი, რომელიც ითვალისწინებს სეზონურობას. 

SARIMAX მოდელი გამოიყen ARIMA-ს 'გაუმჯობესებული' ვერსია, რომელიც ითვალისწინებს სეზონურ კომპონენტებს და გარე ცვლადებს, როგორიცაა Temperature, CPI, Unemployment და MarkDown-ები. მოდელის პარამეტრები (p, d, q) და სეზონური პარამეტრები (P, D, Q, m) ავტომატურად განისაზღვრა pmdarima.auto_arima-ს მეშვეობით, სადაც ( m = 52 ) (წლიური სეზონურობა, რადგან მონაცემები კვირეულია).

პარამეტრების ავტომატური შერჩევა-  ყოველი მაღაზიისთვის გამოვიყენეთ auto_arima, რათა მიეღწა ოპტიმალური (p, d, q) და (P, D, Q, m) პარამეტრები. ( max_p = 2 ), ( max_q = 2 ), ( max_P = 1 ), ( max_Q = 1 ) და ( m = 52 ) განისაზღვრა სეზონური პერიოდის საფუძველზე. 

<img width="979" height="447" alt="image" src="https://github.com/user-attachments/assets/f2eaa4c6-dba3-464a-a9a6-bbd94ad29fa4" />

## N-BEATS 
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
ბლოკების მუშაობის პრინციპი
თითოეული ბლოკი ქმნის ორ კომპონენტს:

Forecast: მომავლის h კვირის პროგნოზი
Backcast: წარსულის რეკონსტრუქცია

საბოლოო forecast = ყველა ბლოკის forecast-ების ჯამი
    - მოდელისთვის საჭირო სვეტები: 
    
    - unique_id - ღირების მაგალითი: "1_Store_Dept_5", "2_Store_Dept_12"
    
    - ds - თარიღი (datetime ფორმატში)
    
    - y - მიზნობრივი ცვლადი (Weekly_Sales)
    
    - IsHoliday - დღესასწაულის ინდიკატორი 

მონაცემების გაყოფა ხდება ასე:
80/20 Temporal Split- არ არის Random Split, რაც სწორია Time Series-ისთვის!
ანუ ვატრენინგებთ წარსულზე, რაღაც თარიღამდე და ვტესტავთ ამ თარიღის მერე.

ჰიპერპარამეტრის ტუნინგის შედეგად საუკეთესო კონფიგურაცია:
NBEATS(

    input_size=52,                               # წინა 52 კვირის მონაცემები
    
    h=40,                                        # 40 კვირიანი პროგნოზი
    
    stack_types=["identity","trend","seasonality"],  # ბლოკების ტიპები
    
    n_blocks=[3,3,3],                            # თითო stack-ში 3 ბლოკი
    
    n_polynomials=2,                             # პოლინომის ხარისხი trend-ისთვის
    
    n_harmonics=2,                               # კოსინუს-სინუს წყვილები seasonality-ისთვის
    
    learning_rate=0.001,                         # სწავლის სიჩქარე
    
    max_steps=2000,                              # ტრენინგის epochs
    
    batch_size=64                                # batch ზომა
)

R² = 0.9348 
MAPE = 4.10% 
WMAE= 39.1575

<img width="1100" height="815" alt="image" src="https://github.com/user-attachments/assets/52897917-c915-4b5b-b703-6f1828d064c9" />

Actual vs Predicted 
წერტილები ძალიან ახლოს არიან წითელ ხაზთან.ეს ნიშნავს რომ მოდელი ზუსტად ვარაუდობს მნიშვნელობებს.

Residuals vs Predicted 
შეცდომები (residuals) თანაბრად არიან განაწილებული 0-ის გარშემო ,არ ჩანს რაიმე მნიშვნელოვანი პატერნი ან სისტემური მიკერძოება.
ეს მიუთითებს რომ მოდელი მუშაობს კარგად

Residual Distribution 
შეცდომების განაწილება ახლოს არის ნორმალურთან, ცენტრირებულია 0-ზე.

Weighted Error Distribution
შეცდომების უმეტესობა მცირეა (მარცხენა მხარეს კონცენტრირებული).ცოტაზე დიდი შეცდომებია, რაც ნორმალურია.

MLflow Run URL: https://dagshub.com/TamariToradze/ML-Final.mlflow/#/experiments/19/runs/66cc17634ef14ce7813f8b135a3e6ef6

ვაი...
<img width="1020" height="66" alt="image" src="https://github.com/user-attachments/assets/010dbe33-21e5-4155-85b5-f5f591080e93" />



## DLinear
DLinear  არის deep learning-ის ერთ-ერთი მოდელი,რომელიც დაფუძნებულია time series ანალიზზე. მოდელი ითვალისწინებს სეზონურ და 'ტრენდულ' (ტენდენციური კომპონენტები) კომპონენტებს, ასევე გარე ფაქტორებს. ყველა ექსპერიმენტი დაფიქსირდა MLflow-ით Dagshub-ის რეპოზიტორიაში

https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/0/runs/7b23c5e4c9eb43c08c724cf4f448cc9e

https://dagshub.com/jgushiann/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/0

პარამეტრები:
- SEQ_LEN = 4
- PRED_LEN = 2
- BATCH_SIZE = 64
- EPOCHS = 50
- LEARNING_RATE = 0.001
- PATIENCE = 10
- TEST_SIZE = 0.1
- VAL_SIZE = 0.05

თავიდან, train.csv, stores.csv და features.csv გააერთიანდა train_full-ში, ხოლო test.csv - test_full-ში, left join-ის გამოყenებით.
მერჯის შემდეგ წარმოიქმნა ორი იდენტური სვეტი IsHoliday_x და IsHoliday_y, რომელიც გავაერთიანე ერთ ცალკე სვეტად(IsHoliday).
განსხვავებით linar/random forest/xgboost მოდელებისგან, აქ არ წაგვიშლია უარყოფითი Weekly_Sales-ის მნიშვნელობები, თუმცა გავფილტრეთ IQR მეთოდით.

შედეგები:
- MAE: 2430.1650
  - RMSE: 4001.7834
  - MAPE: 10758.0645
  - R2: 0.9666

 <img width="1725" height="522" alt="image" src="https://github.com/user-attachments/assets/a664f6a1-407f-4866-841a-044b04584968" />

 <img width="1719" height="387" alt="image" src="https://github.com/user-attachments/assets/8ef0ef03-d09f-402c-8ea8-48e88003ac11" />







