# ML-store-sales-forecast
Problem Definition:
მონაცემებში წარმოდგენილია ვოლმარტის გაყიდვები 45 სხვადასხვა მაღაზიაში, რომლებიც სხვადასხვა რეგიონებში მდებარეობს.
თითოეულ მაღაზიაში არის რამდენიმე დეპარტამენტი და დავალება არის დეპარტამენტის გაყიდვების პროგნოზირება თითოეულ მაღაზიაში.

ვოლმარტი მთელი წლის განმავლობაში ატარებს სხვადასხვა ტიპის ფასდაკლებებს. 
ეს ფასდაკლებები წინ უსწრებს მნიშვნელოვან დღესასწაულებს, რომელთა შორის არიან: სუპერ ბოული, შრომის დღე, მადლიერების დღე და შობა.(4 ყველაზე დიდი დღესასწაული უკვეთება).

# LightGBM
# data preprocessing
აერთიანებს მონაცემებს 
უმკლავდება დუბლირებულ სვეტებს (მაგ. IsHoliday_x, IsHoliday_y) და ავსებს დაკარგულ მნიშვნელობებს.
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
