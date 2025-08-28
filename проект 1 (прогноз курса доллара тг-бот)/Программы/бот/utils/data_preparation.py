import pandas as pd

def prepare_data(input_file='Доллар табл.xlsx', output_file='df_d2.csv', latest_file='latest_data.csv'):
    # Загрузка данных
    df = pd.read_excel(input_file)
    
    # Преобразование дат, только если столбец не является datetime
    if not pd.api.types.is_datetime64_any_dtype(df['data']):
        df['data'] = pd.to_datetime(df['data'], origin='1899-12-30', unit='D')
    
    df.set_index('data', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # Заполнение пропусков с использованием bfill и interpolate
    df['key interest rate'] = df['key interest rate'].bfill()
    df['key interest rate'] = df['key interest rate'].interpolate(method='linear')
    df['inflation'] = df['inflation'].bfill()
    df['inflation'] = df['inflation'].interpolate(method='linear')
    
    # Удаление ненужных колонок
    df.drop(columns=['nominal', 'name of cur'], inplace=True, errors='ignore')
    
    # Добавление лагов
    for i in range(1, 8):
        df[f'curs usd_lag{i}'] = df['curs usd'].shift(i)
    df.dropna(inplace=True)
    
    # Сохранение данных
    df.to_csv(output_file)
    df.tail(100).to_csv(latest_file)
    
    return df

if __name__ == '__main__':
    prepare_data()