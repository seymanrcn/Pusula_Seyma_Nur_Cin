import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


file_path = "data/data_set.xlsx"
df = pd.read_excel(file_path)

print("### Veri Tipleri ###")
print(df.dtypes)

print("\n### İlk 5 Satır ###")
print(df.head())

print("\n### Sayısal Sütunların Özeti ###")
print(df.describe())

# Kategorik Sütunların Özeti
print("\n### Kategorik Sütunların Özetleri ###")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

#Tedavi ve Uygulama Sürelerini Sayısal Hale Getirme
df['TedaviSuresi'] = df['TedaviSuresi'].astype(str).str.extract(r'(\d+)').astype(float)
df['UygulamaSuresi'] = df['UygulamaSuresi'].astype(str).str.extract(r'(\d+)').astype(float)

df.replace(["", "nan", "NaN"], np.nan, inplace=True)

# Eksik değerleri kontrol
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
print("\n### Eksik Değerler ###")
print(missing_df)


# Sütunların kategorik ve sayısal olarak ayrılması
categorical_cols = ['Cinsiyet', 'KanGrubu', 'Bolum', 'Alerji', 'KronikHastalik', 'Tanilar', 'UygulamaYerleri']
numerical_cols = ['Yas', 'TedaviSuresi', 'UygulamaSuresi']

print("\nKategorik sütunlar:", categorical_cols)
print("Sayısal sütunlar:", numerical_cols)

unique_patients = df['HastaNo'].nunique()
print("Eşsiz hasta sayısı:", unique_patients)

sns.set_theme(style="whitegrid", palette="Set2")

# Tüm sütunları aynı olan duplicate satırların çıkarılması
df = df.drop_duplicates(keep='first')

# Tedavi ve uygulama süresi dağılımı (tüm kayıtlar)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['TedaviSuresi'], kde=True, bins=20, color='skyblue')
plt.title('Tedavi Süresi Dağılımı (Tüm Kayıtlar)')

plt.subplot(1,2,2)
sns.histplot(df['UygulamaSuresi'], kde=True, bins=20, color='salmon')
plt.title('Uygulama Süresi Dağılımı (Tüm Kayıtlar)')

plt.tight_layout()
plt.show()

# Hasta bazlı veri
df_unique_patients = df.drop_duplicates(subset=['HastaNo'])

# Yaş dağılımı
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df_unique_patients['Yas'], kde=True, bins=20, color='skyblue')
plt.title('Hasta Bazlı Yaş Dağılımı - Histogram')

plt.subplot(1,2,2)
sns.boxplot(data=df_unique_patients, y='Yas', color='salmon')
plt.title('Hasta Bazlı Yaş Dağılımı - Boxplot')

plt.tight_layout()
plt.show()

# Uyruk dağılımı
plt.figure(figsize=(8,5))
sns.countplot(data=df_unique_patients, x='Uyruk')
plt.title('Hasta Bazlı Uyruk Dağılımı')
plt.xticks(rotation=45)
plt.show()

# Hasta bazlı eşsiz verinin alınması
df_unique_patients = df.drop_duplicates(subset=['HastaNo'])

#pipelinenın sütunlara uygulanması

# veri okunması
file_path = "data/data_set.xlsx"
df = pd.read_excel(file_path)

#  Sayısal sütunları float'a çevrilmesi
numerical_cols = ['Yas', 'TedaviSuresi', 'UygulamaSuresi']
categorical_cols = ['Cinsiyet', 'KanGrubu']

for col in ['TedaviSuresi', 'UygulamaSuresi']:
    df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)


#  Cinsiyet için özel fonksiyonlar
def fill_gender_mode(X):
    X = pd.DataFrame(X, columns=['Cinsiyet'])
    most_common = X['Cinsiyet'].mode()[0]
    X['Cinsiyet'] = X['Cinsiyet'].fillna(most_common)
    return X


def lower_gender(X):
    X = pd.DataFrame(X, columns=['Cinsiyet'])
    X['Cinsiyet'] = X['Cinsiyet'].str.lower()
    return X


def map_gender(X):
    X = pd.DataFrame(X, columns=['Cinsiyet'])
    X['Cinsiyet'] = X['Cinsiyet'].map({'erkek': 0, 'kadın': 1})
    return X


#kan grubu
def fill_blood_by_patient(X, df_original):
    X = pd.DataFrame(X, columns=['KanGrubu'])
    # HastaNo bazlı ffill / bfill
    X['KanGrubu'] = df_original.groupby('HastaNo')['KanGrubu'].transform(lambda x: x.ffill().bfill())
    return X

def encode_and_impute_blood(X, df_numeric):
    X = pd.DataFrame(X, columns=['KanGrubu'])

    # Label encode için önce boşların temizlenmesi
    le = LabelEncoder()
    le.fit(X['KanGrubu'].dropna())

    # Encode, nan değerler korunur
    X['KanGrubu_Label'] = X['KanGrubu'].apply(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)

    # KNN imputer ile kalan boşları doldur
    imputer_array = KNNImputer(n_neighbors=3, weights='distance').fit_transform(
        pd.concat([df_numeric_df, X[['KanGrubu_Label']]], axis=1)
    )
    X['KanGrubu_Label'] = imputer_array[:, -1].round().astype(int)

    # Label decode ile orijinal isimleri geri al
    X['KanGrubu'] = le.inverse_transform(X['KanGrubu_Label'])

    X.drop(columns=['KanGrubu_Label'], inplace=True)
    return X


# ColumnTransformer

df_numeric_df = df[numerical_cols]  # pipeline içinde imputer için kullanılacak

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

gender_transformer = Pipeline(steps=[
    ('fill', FunctionTransformer(fill_gender_mode)),
    ('lower', FunctionTransformer(lower_gender)),
    ('map', FunctionTransformer(map_gender))
])

blood_transformer = Pipeline(steps=[
    ('fill_patient', FunctionTransformer(fill_blood_by_patient, kw_args={'df_original': df})),
    ('encode_impute', FunctionTransformer(encode_and_impute_blood, kw_args={'df_numeric': numerical_cols}))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('gender', gender_transformer, ['Cinsiyet']),
        ('blood', blood_transformer, ['KanGrubu'])
    ],
    remainder='passthrough'
)

# Pipeline

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
df_processed_array = pipeline.fit_transform(df)

# DataFrame'e çevrilmesi

processed_cols = numerical_cols + ['Cinsiyet', 'KanGrubu'] + \
                 [col for col in df.columns if col not in numerical_cols + ['Cinsiyet', 'KanGrubu']]
df_processed = pd.DataFrame(df_processed_array, columns=processed_cols)

# Hasta bazlı eşsiz veri

df_unique_patients = df_processed.drop_duplicates(subset=['HastaNo'])

#  Görselleştirme

sns.set_theme(style="whitegrid", palette="Set2")

# Cinsiyet
plt.figure(figsize=(8, 5))
sns.countplot(data=df_unique_patients, x='Cinsiyet')
plt.xticks(ticks=[0, 1], labels=['erkek', 'kadın'])
plt.title('Hasta Bazlı Cinsiyet Dağılımı (Eksikler Dolduruldu)')
plt.show()

# Kan grubu
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_unique_patients,
    x='KanGrubu',
    order=df_unique_patients['KanGrubu'].value_counts().index
)
plt.title("Hasta Bazlı Kan Grubu Dağılımı (Tüm Eksikler Dolduruldu)")
plt.xlabel("Kan Grubu")
plt.ylabel("Hasta Sayısı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#kronik hastalıklar
from collections import Counter
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer

# Kronik Hastalık Fonksiyonları
def clean_chronic(X):
    # String düzeltmeleri
    X = X.fillna('')
    X = X.str.replace('hiportiroidizm', 'Hipotiroidizm', case=False)
    X = X.str.replace('hipotirodizm', 'Hipotiroidizm', case=False)
    return X

def fill_by_department_pipeline(X, df_full):
    # X: KronikHastalik serisi
    filled = []
    for idx, val in X.items():
        if val == '' or pd.isna(val):
            dept = df_full.loc[idx, 'Bolum']
            most_common = df_full[df_full['Bolum'] == dept]['KronikHastalik'].mode()
            if not most_common.empty:
                filled.append(most_common[0])
            else:
                filled.append('Bilinmiyor')
        else:
            filled.append(val)
    return pd.Series(filled, index=X.index)

def to_list(X):
    # Stringi listeye çevir
    return X.str.split(',').apply(lambda x: [h.strip() for h in x])

# Pipeline Oluştur
chronic_pipeline = Pipeline([
    ('clean', FunctionTransformer(clean_chronic)),
    ('fill', FunctionTransformer(fill_by_department_pipeline, kw_args={'df_full': df})),
    ('to_list', FunctionTransformer(to_list))
])

# Pipeline'ı uygula
df['HastalikListesi'] = chronic_pipeline.fit_transform(df['KronikHastalik'])
df_processed['HastalikListesi'] = df['HastalikListesi']

# One-hot encoding
mlb = MultiLabelBinarizer()
one_hot = mlb.fit_transform(df['HastalikListesi'])
df_hastalik_oh = pd.DataFrame(one_hot, columns=mlb.classes_, index=df.index)
df_processed = pd.concat([df_processed, df_hastalik_oh], axis=1)


# Hasta bazlı kronik hastalık dağılımı (düzgün)

patient_counter = Counter()
for patient, group in df_processed.groupby('HastaNo'):
    unique_diseases = set()
    for diseases in group['HastalikListesi']:
        unique_diseases.update(diseases)
    # Her hastalık her hasta için sadece 1 kez sayılır
    for disease in unique_diseases:
        patient_counter[disease] += 1

patient_counts = pd.Series(patient_counter).sort_values(ascending=False)

# Görselleştirme
plt.figure(figsize=(12,6))
patient_counts.plot(kind='bar', color='skyblue')
plt.title("Kronik Hastalıkların Hasta Bazlı Dağılımı")
plt.xlabel("Kronik Hastalık")
plt.ylabel("Hasta Sayısı")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# df_processed üzerinden sadece gerekli sütunları alınması
age_df = df_processed[['Yas', 'HastalikListesi']].copy()

# Kronik hastalıkları ayrı satırlara açılması
age_expanded = age_df.explode('HastalikListesi')

plt.figure(figsize=(12,6))
sns.boxplot(data=age_expanded, x='HastalikListesi', y='Yas')
plt.xticks(rotation=45, ha='right')
plt.title("Kronik Hastalıklar ve Yaş Dağılımı")
plt.xlabel("Kronik Hastalık")
plt.ylabel("Yaş")
plt.tight_layout()
plt.show()

#BÖLÜM

def fill_department_pipeline(X, df_full):
    filled = []
    for idx, val in X.items():
        if pd.isna(val):
            tedavi = str(df_full.loc[idx, 'TedaviAdi'])
            base_tedavi = tedavi.split('+')[0].strip()
            possible_departments = df_full[df_full['TedaviAdi'].str.contains(base_tedavi, case=False, na=False)]['Bolum'].dropna()
            if not possible_departments.empty:
                filled.append(possible_departments.mode()[0])
            else:
                filled.append('Bilinmiyor')
        else:
            filled.append(val)
    return pd.Series(filled, index=X.index)

def one_hot_department(X):
    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(X.values.reshape(-1,1))
    df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['Bolum']), index=X.index)
    return df_encoded

# Pipeline Oluştur
department_pipeline = Pipeline([
    ('fill', FunctionTransformer(fill_department_pipeline, kw_args={'df_full': df})),
    ('one_hot', FunctionTransformer(one_hot_department))
])


# Pipeline'ı uygula
df_bolum_encoded = department_pipeline.fit_transform(df_processed['Bolum'])
df_processed = pd.concat([df_processed, df_bolum_encoded], axis=1)


# Görselleştirme
# Her hastanın gittiği bölümleri benzersiz hale getir
unique_patient_departments = df_processed.groupby("HastaNo")["Bolum"].unique()

# Bölüm bazlı kaç hasta gitmiş?
department_patient_counts = {}
for bolum_list in unique_patient_departments:
    for bolum in bolum_list:
        if pd.notna(bolum):
            department_patient_counts[bolum] = department_patient_counts.get(bolum, 0) + 1

plt.figure(figsize=(8,5))
plt.bar(department_patient_counts.keys(), department_patient_counts.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Hasta Sayısı")
plt.title("Bölümlere Göre Hasta Dağılımı (Hasta bazlı)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
# Her bölüm için ortalama tedavi süresi
bolum_avg = df_processed.groupby('Bolum')['TedaviSuresi'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
plt.bar(bolum_avg.index, bolum_avg.values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Ortalama Tedavi Süresi (gün)")
plt.title("Bölümlere Göre Ortalama Tedavi Süresi")
plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(12,6))
sns.boxplot(data=df_processed, x='Bolum', y='TedaviSuresi')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Tedavi Süresi (gün)")
plt.title("Bölümlere Göre Tedavi Süresi Dağılımı")
plt.tight_layout()
plt.show()

#ALERJİ
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from sklearn.pipeline import Pipeline

normalize_dict = {
    "volteren": "voltaren",
    "voltaren": "voltaren",
    "gri̇pi̇n": "gripin",
    "gripin": "gripin",
    "novalgin": "novalgin",
    "toz": "toz",
    "polen": "polen",
    "sucuk": "sucuk",
    "arveles": "arveles"
}

def normalize_allergies(X):
    # String normalize etme
    X = X.fillna(np.nan)
    return X.apply(lambda text: ",".join([normalize_dict.get(p.strip(), p.strip())
                                         for p in text.lower().split(",")]) if pd.notna(text) else np.nan)

def fill_allergies(X):
    # Boşları en sık görülen ile doldur
    X = X.replace('', np.nan)
    imputer = SimpleImputer(strategy='most_frequent')  # en sık görüleni kullan
    return pd.Series(imputer.fit_transform(X.to_frame()).ravel(), index=X.index)


def to_set(X):
    # Stringi set hâline getir
    return X.apply(lambda text: set([p.strip() for p in text.split(",")]) if pd.notna(text) else set())

# Pipeline Oluştur
allergy_pipeline = Pipeline([
    ('normalize', FunctionTransformer(normalize_allergies)),
    ('fill', FunctionTransformer(fill_allergies)),
    ('to_set', FunctionTransformer(to_set))
])

# Pipeline'ı uygula
df_processed['Alerji_Set'] = allergy_pipeline.fit_transform(df['Alerji'])

# MultiLabelBinarizer ile one-hot encoding
mlb = MultiLabelBinarizer()
one_hot = mlb.fit_transform(df_processed['Alerji_Set'])  # df_processed kullan
df_allergy_oh = pd.DataFrame(one_hot, columns=mlb.classes_, index=df_processed.index)

# df_processed içinde birleştir
df_processed = pd.concat([df_processed, df_allergy_oh], axis=1)

# Hasta bazlı alerji sayımı
allergy_counts = defaultdict(set)
for _, row in df_processed.iterrows():  # df_processed kullan
    hasta_no = row['HastaNo']
    # Burada kontrol et: Alerji_Set pandas Series mi? Değilse set() olarak çevir
    allergies = row['Alerji_Set'] if isinstance(row['Alerji_Set'], set) else set(row['Alerji_Set'])
    for allergy in allergies:
        if allergy != "Bilinmiyor":
            allergy_counts[allergy].add(hasta_no)

allergy_patient_counts = {k: len(v) for k, v in allergy_counts.items()}

# Görselleştirme
allergy_df = pd.DataFrame({
    'Alerji': list(allergy_patient_counts.keys()),
    'HastaSayisi': list(allergy_patient_counts.values())
}).sort_values('HastaSayisi', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=allergy_df, x='HastaSayisi', y='Alerji', palette='viridis')
plt.title('Her Alerji İçin Benzersiz Hasta Sayısı')
plt.xlabel('Benzersiz Hasta Sayısı')
plt.ylabel('Alerji Tipi')
plt.tight_layout()
plt.show()

# Örnek hasta kontrolü
hasta_no = 145145
hasta_allergy = df_processed[df_processed['HastaNo'] == hasta_no]['Alerji_Set'].iloc[0]
print(f"Hasta {hasta_no} alerjileri:", hasta_allergy)

#tedavi adı
import re
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from rapidfuzz import process, fuzz

# Temizleme ve normalize et
def turkce_lower(metin):
    return metin.replace("İ", "i").replace("I", "ı").lower()

manual_mapping = {
    r'^el rehabilitasyonu(\s+el rehabilitasyonu)+$': 'el rehabilitasyonu',
    r'\b(öçb|acl)\b': 'ön çapraz bağ',
    r'\bptr\b': 'patellar tendon',
    r'\breh\.?\b': 'rehabilitasyonu',
    r'\breha\b': 'rehabilitasyonu',
    r'\bftr\b': 'rehabilitasyonu',
    r'\bop\.?\s*lusu\b': 'operasyon sonrası',
    r'\bopl\b': 'operasyon sonrası',
    r'\bpost\s*op\b': 'operasyon sonrası',
    r'\bpre\s*op\b': 'operasyon öncesi',
    r'\bsend\b': 'sendromu',
    'ağrisi': 'ağrısı',
    'agrisi': 'ağrısı',
    'impingiment': 'impingement',
    'impimgement': 'impingement',
    'adezif kapsülüt': 'adezif kapsülit',
    'bilarteral': 'bilateral',
    'bılateral': 'bilateral',
    'ravmatik': 'travmatik',
    'muskuler': 'muscular',
    'myodascial': 'miyofasiyal',
    'ağrsıı': 'ağrısı',
    r'\bttravmatik\b': 'travmatik',
    r'\b(implant|protez)\b': 'protez',
    'el bilek ağrıs': 'el bilek ağrısı',
    r'el rehabilitasyon programı.*': 'el rehabilitasyonu',
    r'dorsalji\s*(bel|boyun|sırt|dorsal|koksiks|servikal)': 'dorsalji',
    r'\başil\s*rüptürü.*': 'aşil rüptürü',
    r'\bdiz\s*implant.*': 'diz protezi',
    r'\bön\s*çapraz\s*bağ.*': 'ön çapraz bağ rehabilitasyonu',
}
gereksiz_kelimeler = ["xx", "onur", "test", "deneme", "nan"]

def temizle_ve_normalize_et(metin):
    t = str(metin)
    t = turkce_lower(t)
    t = re.sub(r'\s*-\s*', ' ', t)
    t = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', t)
    for k in gereksiz_kelimeler:
        t = re.sub(rf"\b{k}\b", "", t)
    t = re.sub(r"\b(sağ|sol|bilateral)\b", "", t)
    for key, value in manual_mapping.items():
        t = re.sub(key, value, t)
    t = re.sub(r'(\w)\s*\.\s*(\w)', r'\1 \2', t)
    t = t.replace("+", ",")
    t = re.sub(r'(\D)(\d+)', r'\1', t)
    t = re.sub(r'\b\d+\b', '', t)
    t = t.strip(" .,-_")
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r',+', ',', t).strip(', ')
    t = t.title()
    t = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', t)
    return t

# Fuzzy Group
def fuzzy_group(X):
    FUZZY_THRESHOLD = 86
    tedavi_listesi = X.unique().tolist()
    unique_norm = []
    mapping = {}
    for item in tedavi_listesi:
        item_norm = item.title()
        if not unique_norm:
            unique_norm.append(item_norm)
            mapping[item] = item_norm
        else:
            match, score, _ = process.extractOne(item_norm, unique_norm, scorer=fuzz.token_sort_ratio)
            if score >= FUZZY_THRESHOLD:
                mapping[item] = match
            else:
                unique_norm.append(item_norm)
                mapping[item] = item_norm
    return X.map(mapping)

# Pipeline
class TedaviPipeline(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df_copy = X.copy()
        df_copy = df_copy.assign(TedaviAdi_split=df_copy['TedaviAdi'].str.split(r'[,\+]')).explode('TedaviAdi_split')
        df_copy['TedaviAdi_clean'] = df_copy['TedaviAdi_split'].apply(temizle_ve_normalize_et)
        df_copy.dropna(subset=['TedaviAdi_clean'], inplace=True)
        df_copy['TedaviAdi_clean'] = fuzzy_group(df_copy['TedaviAdi_clean'])
        # Hasta bazlı doldurma
        df_copy['TedaviAdi_clean'] = df_copy.groupby('HastaNo')['TedaviAdi_clean'].transform(lambda x: x.ffill().bfill())
        return df_copy

tedavi_pipeline = TedaviPipeline()
df_processed = tedavi_pipeline.fit_transform(df)


#TedaviSuresi string -> numeric
def sureyi_sayiya_cevir(x):
    num = re.findall(r'\d+', str(x))
    return int(num[0]) if num else None

df_processed['TedaviSuresi_num'] = df_processed['TedaviSuresi'].apply(sureyi_sayiya_cevir)

#En sık görülen 10 tedavi
top_10_tedaviler = df_processed['TedaviAdi_clean'].value_counts().head(10).index.tolist()

# Ortalama tedavi süresi (sadece top 10 için)
df_top10 = df_processed[df_processed['TedaviAdi_clean'].isin(top_10_tedaviler)]
tedavi_sureleri = df_top10.groupby('TedaviAdi_clean')['TedaviSuresi_num'].mean().sort_values(ascending=False)

# Görselleştirme
plt.figure(figsize=(12,6))
sns.barplot(x=tedavi_sureleri.values, y=tedavi_sureleri.index, palette='magma')
plt.xlabel("Ortalama Tedavi Süresi (Seans)")
plt.ylabel("Tedavi Adı")
plt.title("En Sık Karşılaşılan 10 Tedavi için Ortalama Tedavi Süresi")
plt.tight_layout()
plt.show()


#UYGULAMA YERLERİ
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class UygulamaYerPipelineProcessed(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = None

    def fit(self, X, y=None):
        X_copy = X.copy()
        # Listeye çevir ve temizle
        X_copy['UygulamaYerleri_List'] = X_copy['UygulamaYerleri'].fillna('').str.split(',')
        X_copy['UygulamaYerleri_List'] = X_copy['UygulamaYerleri_List'].apply(lambda x: [y.strip() for y in x if y.strip()])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(X_copy['UygulamaYerleri_List'])
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Hasta + Tedavi bazlı eksik doldurma
        def doldur_hasta_tedavi(grup):
            for tedavi in grup['TedaviAdi_clean'].unique():
                mask = (grup['TedaviAdi_clean'] == tedavi) & (grup['UygulamaYerleri'].isna())
                if mask.any():
                    dolu = grup.loc[(grup['TedaviAdi_clean'] == tedavi) & (grup['UygulamaYerleri'].notna()), 'UygulamaYerleri']
                    if not dolu.empty:
                        grup.loc[mask, 'UygulamaYerleri'] = dolu.iloc[0]
            grup['UygulamaYerleri'] = grup['UygulamaYerleri'].fillna('Bilinmiyor')
            return grup

        X_copy = X_copy.groupby('HastaNo').apply(doldur_hasta_tedavi).reset_index(drop=True)

        # Listeye çevir ve temizle
        X_copy['UygulamaYerleri_List'] = X_copy['UygulamaYerleri'].str.split(',')
        X_copy['UygulamaYerleri_List'] = X_copy['UygulamaYerleri_List'].apply(lambda x: [y.strip() for y in x if y.strip()])

        # One-hot encode
        uyg_oh = pd.DataFrame(
            self.mlb.transform(X_copy['UygulamaYerleri_List']),
            columns=[f"UygYer_{c}" for c in self.mlb.classes_],
            index=X_copy.index
        )

        # Orijinal df_processed ile birleştir
        X_processed_encoded = pd.concat([X_copy, uyg_oh], axis=1)
        return X_processed_encoded

# Kullanım
uyg_pipeline_processed = UygulamaYerPipelineProcessed()
df_processed = uyg_pipeline_processed.fit_transform(df_processed)

# Duplicate satırları HastaNo + TedaviAdi_clean + UygulamaSuresi bazında kaldır
df_processed = df_processed.drop_duplicates(subset=['HastaNo', 'TedaviAdi_clean', 'UygulamaSuresi'])

#Kontrol
hasta_145141 = df_processed[df_processed['HastaNo'] == 145141]
print(hasta_145141[['HastaNo', 'UygulamaYerleri', 'UygulamaSuresi', 'TedaviSuresi', 'TedaviAdi_clean']].to_string(index=False))

#Tanılar
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re
from rapidfuzz import process, fuzz
from sklearn.base import BaseEstimator, TransformerMixin

# VERİYİ OKU
df_processed['HastaNo'] = df_processed['HastaNo'].astype(str)
df_processed['Tanilar'] = df_processed['Tanilar'].astype(str).str.strip().str.lower()

# Ek kelimeler listesi
ek_kelimeler = [
    'diğer','servikotorasik bölge','tanımlanmamış','el','başka yerde sınıflanmamış',
    'diğer tanımlanmış','tanımlanmamış komplikasyonlarla birlikte','lumbosakral bölge',
    'şimdiki','diğer sendromları','birden fazla yer','vertebrada','bilateral',
    'servikal bölge','pelvik bölge ve kalça','torasik bölge','kol','ayak bileği hariç',
    'sakral ve sakrokoksigeal bölge','yeri tanımlanmamış','omuz bölgesi',
    'ayak bileği ve ayak','kapalı','başka yerde sınıflanmış hastalıklarda','bacak'
]

#TANILAR PIPELINE
class TanilarPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ek_kelimeler=None, fuzzy_threshold=85):
        self.ek_kelimeler = ek_kelimeler or []
        self.fuzzy_threshold = fuzzy_threshold
        self.gruplar = None
        self.tedavi_most_common = None

    def temizle(self, metin):
        t = str(metin).lower()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r'\s+', ' ', t).strip()
        for ek in self.ek_kelimeler:
            t = t.replace(ek, '')
        return t.strip()

    def fit(self, X, y=None):
        df_copy = X.copy()
        tedavi_tanilar = df_copy.groupby('TedaviAdi')['Tanilar'].apply(lambda x: x.dropna().tolist())
        self.tedavi_most_common = {}
        temp_all_dict = {}
        for tedavi, tanilar_list in tedavi_tanilar.items():
            temp_all = []
            for t in tanilar_list:
                for t_split in re.split(r'[;,]', str(t)):
                    t_clean = self.temizle(t_split)
                    if t_clean:
                        temp_all.append(t_clean)
            if temp_all:
                self.tedavi_most_common[tedavi] = Counter(temp_all).most_common(1)[0][0]
                temp_all_dict[tedavi] = temp_all

        all_tanilar = [t for t_list in temp_all_dict.values() for t in t_list]

        self.gruplar = {}
        for t in all_tanilar:
            if not self.gruplar:
                self.gruplar[t] = [t]
            else:
                en_yakin, skor, _ = process.extractOne(t, list(self.gruplar.keys()), scorer=fuzz.ratio)
                if skor >= self.fuzzy_threshold:
                    self.gruplar[en_yakin].append(t)
                else:
                    self.gruplar[t] = [t]
        return self

    def transform(self, X):
        df_copy = X.copy()
        hasta_standart = {}
        for idx, row in df_copy.iterrows():
            hasta = row['HastaNo']
            tanilar_raw = row['Tanilar']
            temp_list = []

            if pd.isna(tanilar_raw) or str(tanilar_raw).strip() == '':
                tedavi = row['TedaviAdi']
                temp_list.append(self.tedavi_most_common.get(tedavi, 'bilinmiyor'))
            else:
                for t in re.split(r'[;,]', str(tanilar_raw)):
                    t_clean = self.temizle(t)
                    if t_clean:
                        temp_list.append(t_clean)

            temp_fuzzy = []
            for t in temp_list:
                if t.lower() == 'nan':
                    continue
                en_yakin, skor, _ = process.extractOne(t, list(self.gruplar.keys()), scorer=fuzz.ratio)
                temp_fuzzy.append(en_yakin)
            hasta_standart[hasta] = list(set(temp_fuzzy)) if temp_fuzzy else ['bilinmiyor']

        df_copy['Tanilar_clean'] = df_copy['HastaNo'].map(hasta_standart)
        return df_copy

# Pipeline Uygulama
tanilar_pipeline = TanilarPipeline(ek_kelimeler=ek_kelimeler)
df_processed = tanilar_pipeline.fit_transform(df_processed)

# Frekans hesabı ve en sık 10 tanıyı görselleştir
hasta_tanil_set = df_processed.groupby('HastaNo')['Tanilar_clean'].apply(
    lambda x: set([t for sub in x for t in sub])
).reset_index()

hasta_tanil_set['Tanilar_clean'] = hasta_tanil_set['Tanilar_clean'].apply(lambda x: x if x else {'bilinmiyor'})

tanilar_count = Counter()
for t_list in hasta_tanil_set['Tanilar_clean']:
    for t in t_list:
        tanilar_count[t] += 1

top10_tanilar = tanilar_count.most_common(10)
tanilar, counts = zip(*top10_tanilar)

plt.figure(figsize=(12,6))
sns.barplot(x=list(counts), y=list(tanilar), palette="viridis")
plt.xlabel("Hasta Sayısı")
plt.ylabel("Tanılar")
plt.title("En Sık Karşılaşılan 10 Tanı (Temizlenmiş ve Eksikler Doldurulmuş)")
plt.tight_layout()
plt.show()

# Örnek Hasta Kontrolü
hasta_no = '145141'
hasta_tanil = df_processed[df_processed['HastaNo']==hasta_no]['Tanilar_clean'].iloc[0]
print(f"Hasta {hasta_no} tanıları:", hasta_tanil)

#Yaş ve Tedavi Süresi ilişkisi
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_processed.drop_duplicates(subset=['HastaNo']),
    x='Yas',
    y='TedaviSuresi',
    hue='Cinsiyet'
)
plt.title('Hasta Bazlı Yaş ve Tedavi Süresi İlişkisi')
plt.xlabel('Yaş')
plt.ylabel('Tedavi Süresi')
plt.tight_layout()
plt.show()

# Cinsiyete göre tedavi süresi dağılımı
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Cinsiyet',
    y='TedaviSuresi',
    data=df_processed.drop_duplicates(subset=['HastaNo'])
)
plt.title('Cinsiyete Göre Tedavi Süresi Dağılımı')
plt.tight_layout()
plt.show()
