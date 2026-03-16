# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

# %%
df.head()

# %%
df

# %%
df.shape

# %%
df.describe()

# %%
df.info()

# %%
# Vẽ biểu đồ phân phối của chiều cao và cân nặng
plt.figure(figsize=(8, 8))
sns.scatterplot(x='Height', y='Weight', hue='Index', data=df, palette='Set1')
plt.title('Scatter Plot of Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(title='Index')
plt.show()

# %%
# Scatter plots of Height vs Weight by gender
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

sns.scatterplot( x='Height', y='Weight', hue='Index', data=df[df['Gender'] == 'Female'], palette='Set1', ax=axes[0])
axes[0].set_title('Scatter Plot of Height vs Weight for Female')
axes[0].set_xlabel('Height (cm)')
axes[0].set_ylabel('Weight (kg)')
axes[0].legend(title='Index')

sns.scatterplot(x='Height', y='Weight', hue='Index', data=df[df['Gender'] == 'Male'], palette='Set1', ax=axes[1])
axes[1].set_title('Scatter Plot of Height vs Weight for Male')
axes[1].set_xlabel('Height (cm)')
axes[1].set_ylabel('Weight (kg)')
axes[1].legend(title='Index')

plt.tight_layout()
plt.show()



# %%
#biểu đồ count các Index theo giới tính
plt.figure(figsize=(8, 6))
sns.countplot(x='Index', hue='Gender', data=df, palette='Set1')
plt.title('Count of Index by Gender')
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

# %% [markdown]
# ### Dựa vào phân bố dữ liệu từ biểu đồ Scatter và biểu dồ Index theo Gender, ta có thể thấy phần lớn tập dữ liệu là các index 4 và 5, đây là các nhóm 2 trạng thái Obesity và Extreme Obesity, ngược lại đối với nhóm index 0 1 là Extremely WeakWeak thì rất ít dữ liệu, nhóm 2 3 là Normal và Overweight thì có lượng tương đương nhau => Từ lượng dữ liệu và phân bố các classes, nhóm chúng lại thành 3 nhóm như sau:
# - Nhóm 1 - Gầy: Các index 0 1
# - Nhóm 2 - Bình thường: Các index 2 3 
# - Nhóm 3 - Mập: Các index 4 5 

# %%
#Chuẩn hóa dữ liệu Gender 
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df

# %%
#gộp nhóm Index thành 3 nhóm: 0 1 = Gầy (1 ), 2 3 = Bình thường ( 2 ), Mập (3)  = 4 5 
mapping_dict = {
    0: 'Gầy', 
    1: 'Gầy',
    2: 'Bình thường', 
    3: 'Bình thường',
    4: 'Mập', 
    5: 'Mập'
}
df['Index'] = df['Index'].map(mapping_dict)
df

# %%
df['Index'] = df['Index'].map({'Gầy': 0, 'Bình thường': 1, 'Mập': 2})
df

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Height', y='Weight', hue='Index', data=df, palette='Set1')
plt.title('Scatter Plot of Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(title='Index')
plt.show()

# %%
#phân phối của các biến liên tục
ax, fg = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Height'],bins=20, kde=True, stat='density', color='skyblue', ax=fg[0])
fg[0].set_title('Distribution of Height')
sns.histplot(df['Weight'], bins=20, kde=True, stat='density', color='orange', ax=fg[1])
fg[1].set_title('Distribution of Weight')
plt.tight_layout()
plt.show()

# %%
df.describe()

# %% [markdown]
# ### với số lượng dữ liệu chỉ 500 dòng và không có phân phối đẹp, đồng thời các biểu đồ cũng cho ta thấy rõ rằng đây là 1 tập dữ liệu imbalance dataset ( tập dữ liệu mất cân bằng )
# =>  Vì vậy áp dụng các kỹ thuật xử lý dự liệu như Smote hoặc RandamOversampling để cân bằng dữ liệu

# %%
#áp dụng smote để cân bằng dữ liệu
from imblearn.over_sampling import SMOTE

# %%
X = df[['Gender', 'Height', 'Weight']].values
y = df['Index'].values

# %%
from sklearn.model_selection import train_test_split

# %%
# Sử dụng stratify=y để đảm bảo tỷ lệ các lớp Index trong tập Train và Test là tương đương nhau
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %%
X_train

# %%
X_train.shape

# %%
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
X_train_resampled.shape

# %%
df_resampled = pd.DataFrame(X_train_resampled, columns=['Gender', 'Height', 'Weight'])
df_resampled['Index'] = y_train_resampled

# %%
df_resampled.describe()

# %%
print(f"df_resampled['Index'].value_counts():\n{df_resampled['Index'].value_counts()}")

# %% [markdown]
# Thuật toán Gaussian Naive Bayes from scratch
# $$P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

# %%
def train_gnb(X,y):
    classes = np.unique(y)
    models_parameters = {}
    
    for c in classes:
        X_c = X[y == c]
        mean = X_c.mean(axis=0) # tính trung bình 
        var = X_c.var(axis=0) # tính phương sai 
        prior = X_c.shape[0] / X.shape[0] # tính tiên nghiệm ở lớp C
        models_parameters[c] = {'mean': mean, 'var': var, 'prior': prior}
        #print(f"Class {c}: mean={mean}, var={var}, prior={prior}")
    return classes, models_parameters

# %%
classes, model_params = train_gnb(X_train_resampled, y_train_resampled)

# %% [markdown]
# ### Sau khi tính toán được trung bình và phương sai, tiên nghiệm của các class, ta tiến hình tính toán Likelihood bằng hàm mật độ xác suất
# $$p(x|y = c) = \prod_{j=1}^{D} \mathcal{N}(x_j|\mu_{jc}, \sigma_{jc}^2)$$
# - $x$: Là một vector chứa toàn bộ thông tin của đối tượng mới (VD: $x = [172 \text{ cm}, 68 \text{ kg}]$).
# - $y = c$: Lớp mà ta đang giả định (VD: $c = \text{Nam}$).

# %%
def cal_pdf(X, mean, var):
    numerator = np.exp( -0.5*(X - mean)**2/(2*var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator

# %%
def predict_single_point(X, classes, model_params):
    posteriors = []
    
    for c in classes:
        mean = model_params[c]['mean']
        var = model_params[c]['var']
        prior_log = np.log(model_params[c]['prior'])
        
        likelihood_log = np.sum(np.log(cal_pdf(X, mean, var))) #ép về log để tránh underflow ( tràn số dưới )
        
        #tính toán hậu nghiệm
        posterior_log = prior_log + likelihood_log
        posteriors.append(posterior_log)
        #print(f"Class {c}: Prior Log={prior_log:.4f}, Likelihood Log={likelihood_log:.4f}, Posterior Log={posterior_log:.4f}")
    #trả về lớp có hậu nghiệm cao nhất 
    return classes[np.argmax(posteriors)]


# %%
#dự đoán cho toàn tập dữ liệu
def predict(X, classes, model_params):
    y_pred = np.array([predict_single_point(x, classes, model_params) for x in X])
    return np.array(y_pred)

# %%
y_pred = predict(X_test, classes, model_params)

# %%
y_pred

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# %%
print(classification_report(y_test, y_pred))

# %%
cfs_mt = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
label = ['0: Gầy', '1: Bình thường', '2: Mập']
sns.heatmap(cfs_mt, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# %%
X_testing = np.array([[0, 160, 50]]) # Giới tính: Nam (0), Chiều cao: 160 cm, Cân nặng: 50 kg
predict_single_point(X_testing, classes, model_params)

# %% [markdown]
# ### Thử nghiệm lại trên Gaussian Naive Bayes của thư viện Sklearn

# %%
from sklearn.naive_bayes import GaussianNB

# %%
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)

# %%
import pickle
with open('gnb_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# %%
# accuracy của sklearn
y_pred_sklearn = model.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

# %%
acc_sklearn

# %%
# confusion matrix mô hình của sklearn
cfs_mt_sklearn = confusion_matrix(y_test, y_pred_sklearn)
print("Confusion Matrix (sklearn):")   
label = ['0: Gầy', '1: Bình thường', '2: Mập']
sns.heatmap(cfs_mt_sklearn, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (sklearn)')
plt.show()

# %%
X_test

# %%
y_test

# %%
X_1 = np.array([[0, 140, 11]])
model.predict(X_1)

# %%



