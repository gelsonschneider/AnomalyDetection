import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv'
OUTPUT_FILE = 'inpatient_claims_sample1_model_features_V_EN.csv'

df = pd.read_csv(INPUT_FILE)
print(df.head())

# Check data types
print(df.dtypes.value_counts())

# Check for null values
nulos = df.isnull().sum()
print(nulos[nulos > 0])

def processar_datas(df):
    """
    Processes date variables and creates a length of stay (LOS) feature
    """
    # Convert date columns to datetime
    colunas_data = ['CLM_FROM_DT', 'CLM_THRU_DT', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT']
    
    for col in colunas_data:
        if col in df.columns:
            # Convert to a string, then to a datetime
            df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
    
    # Create feature: Length of Stay (LOS)
    # Using admission date and discharge date
    if 'CLM_ADMSN_DT' in df.columns and 'NCH_BENE_DSCHRG_DT' in df.columns:
        df['LOS'] = (df['NCH_BENE_DSCHRG_DT'] - df['CLM_ADMSN_DT']).dt.days
        
        # For cases where LOS is negative or zero, use CLM_UTLZTN_DAY_CNT
        mask_los_invalido = (df['LOS'].isna()) | (df['LOS'] < 0)
        if 'CLM_UTLZTN_DAY_CNT' in df.columns:
            df.loc[mask_los_invalido, 'LOS'] = df.loc[mask_los_invalido, 'CLM_UTLZTN_DAY_CNT']
        
        # Fill in remaining null values with 0 or the median
        df['LOS'] = df['LOS'].fillna(0)
        
        # Limit extreme values (e.g., hospitalizations lasting more than 365 days are rare)
        df['LOS'] = df['LOS'].clip(0, 365)
        

    # Remove the original date columns (which are no longer needed)
    colunas_para_remover = ['CLM_FROM_DT', 'CLM_THRU_DT', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT']
    for col in colunas_para_remover:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df

def processar_identificadores(df):
    colunas_remover = ['DESYNpUF_ID', 'CLM_ID', 'SEGMENT']
    colunas_existentes = [col for col in colunas_remover if col in df.columns]
    df = df.drop(colunas_existentes, axis=1)
    
    print(f"  - Removed columns: {colunas_existentes}")
    return df

def processar_provedores(df):

    colunas_provedor = ['PRVDR_NUM', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI']
    colunas_existentes = [col for col in colunas_provedor if col in df.columns]
    df = df.drop(colunas_existentes, axis=1)
    
    print(f"  -  Removed columns: {colunas_existentes}")
    return df

def processar_diagnosticos(df):

    # Collect all diagnostic columns
    colunas_diagnostico = [col for col in df.columns if col.startswith('ICD9_DGNS_CD_')]
    
    if colunas_diagnostico:
        # Criar feature: número de diagnósticos na claim
        df['NUM_DIAGNOSTICOS'] = df[colunas_diagnostico].notna().sum(axis=1)
        
        # Criar feature: diagnóstico principal (primeiro) e secundário (segundo)
        if 'ICD9_DGNS_CD_1' in df.columns:
            df['DIAG_PRINCIPAL'] = df['ICD9_DGNS_CD_1'].astype(str)
            # Manter apenas o código principal (3 primeiros caracteres para ICD-9)
            df['DIAG_PRINCIPAL_CATEGORIA'] = df['DIAG_PRINCIPAL'].str[:3]
        
        if 'ICD9_DGNS_CD_2' in df.columns:
            df['DIAG_SECUNDARIO'] = df['ICD9_DGNS_CD_2'].astype(str)
        
        print(f"  - {len(colunas_diagnostico)} colunas de diagnóstico processadas")
        print(f"  - Média de diagnósticos por claim: {df['NUM_DIAGNOSTICOS'].mean():.2f}")
    
    return df

def processar_procedimentos(df):

    # Coletar todas as colunas de procedimento
    colunas_procedimento = [col for col in df.columns if col.startswith('ICD9_PRCDR_CD_')]
    
    if colunas_procedimento:
        # Criar feature: número de procedimentos na claim
        df['NUM_PROCEDIMENTOS'] = df[colunas_procedimento].notna().sum(axis=1)
        
        # Criar feature: procedimento principal (primeiro)
        if 'ICD9_PRCDR_CD_1' in df.columns:
            df['PROC_PRINCIPAL'] = df['ICD9_PRCDR_CD_1'].astype(str)
            # Manter apenas o código principal (3 primeiros caracteres para ICD-9)
            df['PROC_PRINCIPAL_CATEGORIA'] = df['PROC_PRINCIPAL'].str[:3]
        
        print(f"  - {len(colunas_procedimento)} colunas de procedimento processadas")
        print(f"  - Média de procedimentos por claim: {df['NUM_PROCEDIMENTOS'].mean():.2f}")
    
    return df

def processar_hcpcs(df):

    colunas_hcpcs = [col for col in df.columns if col.startswith('HCPCS_CD_')]
    
    if colunas_hcpcs:
        # Contar quantos HCPCS estão preenchidos por claim
        df['NUM_HCPCS'] = df[colunas_hcpcs].notna().sum(axis=1)
        
        # Remover colunas HCPCS individuais (muito esparsas)
        df = df.drop(colunas_hcpcs, axis=1)
        
        print(f"  - {len(colunas_hcpcs)} colunas HCPCS removidas")
        print(f"  - Média de HCPCS por claim: {df['NUM_HCPCS'].mean():.2f}")
    
    return df

def processar_financeiro(df):
    
    # Garantir que colunas financeiras sejam numéricas
    colunas_financeiras = [
        'CLM_PMT_AMT', 
        'NCH_PRMKY_PYR_CLM_PD_AMT',
        'CLM_PASS_THRU_PER_DIEM_AMT',
        'NCH_BENE_IP_DDCTBL_AMT',
        'NCH_BENE_PTA_COINSRNC_LBLTY_AM',
        'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM'
    ]
    
    for col in colunas_financeiras:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Criar feature: Custo Total da Internação
    df['CUSTO_TOTAL'] = (
        df['CLM_PMT_AMT'] + 
        df['NCH_PRMKY_PYR_CLM_PD_AMT'] +
        df['NCH_BENE_IP_DDCTBL_AMT'] +
        df['NCH_BENE_PTA_COINSRNC_LBLTY_AM'] +
        df['NCH_BENE_BLOOD_DDCTBL_LBLTY_AM']
    )
    
    # Criar feature: Custo por Dia
    df['CUSTO_POR_DIA'] = df['CUSTO_TOTAL'] / (df['LOS'] + 1)  # +1 para evitar divisão por zero
    
    # Criar feature: Proporção paga pelo Medicare
    df['PROPORCAO_MEDICARE'] = df['CLM_PMT_AMT'] / (df['CUSTO_TOTAL'] + 1)
    df['PROPORCAO_MEDICARE'] = df['PROPORCAO_MEDICARE'].clip(0, 1)
    
    # Criar feature: Flag de pagamento zero
    df['FLAG_PAGAMENTO_ZERO'] = (df['CLM_PMT_AMT'] == 0).astype(int)
    
    # Tratar outliers financeiros (usando IQR)
    for col in ['CLM_PMT_AMT', 'CUSTO_TOTAL']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 3 * IQR  # Usar 3*IQR (menos agressivo que 1.5*IQR)
            df[f'{col}_CAP'] = df[col].clip(upper=limite_superior)
            
            # Flag de outlier financeiro
            df[f'{col}_FLAG_OUTLIER'] = (df[col] > limite_superior).astype(int)
    
    print(f"  - Features financeiras criadas e processadas")
    print(f"  - Custo total médio: ${df['CUSTO_TOTAL'].mean():.2f}")
    print(f"  - Custo total mediano: ${df['CUSTO_TOTAL'].median():.2f}")
    
    return df

def processar_drg(df):
    """
    Processa código DRG
    """
    print("\nProcessando DRG...")
    
    if 'CLM_DRG_CD' in df.columns:
        # Converter para string e limpar
        df['CLM_DRG_CD'] = df['CLM_DRG_CD'].astype(str).str.strip()
        
        # Criar categorias de DRG (principais grupos)
        # DRG MDC (Major Diagnostic Category) - primeiros 1-2 dígitos
        df['MDC'] = df['CLM_DRG_CD'].str[:2]
        
        # Flag de DRG válido
        df['FLAG_DRG_VALIDO'] = (~df['CLM_DRG_CD'].isin(['nan', 'None', ''])).astype(int)
        
        print(f"  - DRG processado. {df['FLAG_DRG_VALIDO'].sum()} DRGs válidos")
    
    return df

def processar_diagnostico_admissao(df):
 
    if 'ADMTNG_ICD9_DGNS_CD' in df.columns:
        df['ADMTNG_ICD9_DGNS_CD'] = df['ADMTNG_ICD9_DGNS_CD'].astype(str).str.strip()
        
        # Criar categoria do diagnóstico de admissão (primeiros 3 caracteres)
        df['ADMTNG_CATEGORIA'] = df['ADMTNG_ICD9_DGNS_CD'].str[:3]
        
        # Flag de diagnóstico válido
        df['FLAG_DIAG_ADM_VALIDO'] = (~df['ADMTNG_ICD9_DGNS_CD'].isin(['nan', 'None', ''])).astype(int)
        
        print(f"  - Diagnóstico de admissão processado")
    
    return df

def processar_utilizacao(df):

    if 'CLM_UTLZTN_DAY_CNT' in df.columns:
        df['CLM_UTLZTN_DAY_CNT'] = pd.to_numeric(df['CLM_UTLZTN_DAY_CNT'], errors='coerce').fillna(0)
        df['CLM_UTLZTN_DAY_CNT'] = df['CLM_UTLZTN_DAY_CNT'].clip(0, 365)
        
        print(f"  - Dias de utilização processados")
    
    return df

def criar_features_complexas(df):

    # Feature: Complexidade do caso (combinação de diagnósticos e procedimentos)
    if 'NUM_DIAGNOSTICOS' in df.columns and 'NUM_PROCEDIMENTOS' in df.columns:
        df['COMPLEXIDADE'] = df['NUM_DIAGNOSTICOS'] + df['NUM_PROCEDIMENTOS']
    
    # Feature: Eficiência (custo por dia de internação)
    if 'CUSTO_TOTAL' in df.columns and 'LOS' in df.columns:
        df['EFICIENCIA'] = df['CUSTO_TOTAL'] / (df['LOS'] + 1)
    
    # Feature: Paciente de alto custo (flag)
    if 'CUSTO_TOTAL' in df.columns:
        percentil_90 = df['CUSTO_TOTAL'].quantile(0.90)
        df['FLAG_ALTO_CUSTO'] = (df['CUSTO_TOTAL'] > percentil_90).astype(int)
    
    # Feature: Internação longa (flag)
    if 'LOS' in df.columns:
        percentil_90_los = df['LOS'].quantile(0.90)
        df['FLAG_INTERNACAO_LONGA'] = (df['LOS'] > percentil_90_los).astype(int)
    
    return df

# Apply all preprocessing functions
df = processar_identificadores(df)
df = processar_provedores(df)
df = processar_datas(df)
df = processar_diagnostico_admissao(df)
df = processar_drg(df)
df = processar_diagnosticos(df)
df = processar_procedimentos(df)
df = processar_hcpcs(df)
df = processar_utilizacao(df)
df = processar_financeiro(df)
df = criar_features_complexas(df)


# Estatísticas das principais features
print("\nEstatísticas das principais features:")
features_principais = ['LOS', 'CLM_PMT_AMT', 'CUSTO_TOTAL', 'NUM_DIAGNOSTICOS', 'NUM_PROCEDIMENTOS']
features_existentes = [col for col in features_principais if col in df.columns]
print(df[features_existentes].describe())

print("\nDistribuição de DRG (top 10):")
if 'CLM_DRG_CD' in df.columns:
    print(df['CLM_DRG_CD'].value_counts().head(10))

print("\nVerificando valores nulos nas colunas finais:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Salvar arquivo processado
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nArquivo processado salvo como: {OUTPUT_FILE}")
print(f"Tamanho final: {df.shape}")

# Salvar também uma versão com as features mais relevantes para modelagem
features_modelo = [
    'LOS', 'CLM_PMT_AMT', 'CUSTO_TOTAL', 'CUSTO_POR_DIA', 
    'NUM_DIAGNOSTICOS', 'NUM_PROCEDIMENTOS', 'NUM_HCPCS',
    'PROPORCAO_MEDICARE', 'EFICIENCIA', 'COMPLEXIDADE',
    'FLAG_PAGAMENTO_ZERO', 'FLAG_ALTO_CUSTO', 'FLAG_INTERNACAO_LONGA'
]

# Adicionar variáveis categóricas relevantes
features_categoricas = ['CLM_DRG_CD', 'MDC', 'ADMTNG_CATEGORIA', 'DIAG_PRINCIPAL_CATEGORIA']
features_categoricas_existentes = [col for col in features_categoricas if col in df.columns]

features_modelo.extend(features_categoricas_existentes)

# Selecionar apenas features existentes
features_modelo = [col for col in features_modelo if col in df.columns]

df_modelo = df[features_modelo]
df_modelo.to_csv('inpatient_claims_sample1_model_features.csv', index=False)
print(f"Arquivo com features para modelagem salvo como: inpatient_claims_sample1_model_features.csv")
