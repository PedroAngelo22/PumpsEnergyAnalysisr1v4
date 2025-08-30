import streamlit as st
import pandas as pd
import math
import time
import numpy as np
from scipy.optimize import root
import graphviz
import matplotlib.pyplot as plt
import io

# Importações para ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.colors import HexColor

# Configura o Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')

# --- Cores do PDF ---
PRIMARY_COLOR = HexColor('#004173') # Azul executivo

# --- BIBLIOTECAS DE DADOS (K_FACTORS, FLUIDOS, MATERIAIS) ---
MATERIAIS = {
    "Aço Carbono (novo)": 0.046, "Aço Carbono (pouco uso)": 0.1, "Aço Carbono (enferrujado)": 0.2,
    "Aço Inox": 0.002, "Ferro Fundido": 0.26, "PVC / Plástico": 0.0015, "Concreto": 0.5
}
K_FACTORS = {
    "Entrada de Borda Viva": 0.5, "Entrada Levemente Arredondada": 0.2, "Entrada Bem Arredondada": 0.04,
    "Saída de Tubulação": 1.0, "Válvula Gaveta (Totalmente Aberta)": 0.2, "Válvula Gaveta (1/2 Aberta)": 5.6,
    "Válvula Globo (Totalmente Aberta)": 10.0, "Válvula de Retenção (Tipo Portinhola)": 2.5,
    "Cotovelo 90° (Raio Longo)": 0.6, "Cotovelo 90° (Raio Curto)": 0.9, "Cotovelo 45°": 0.4,
    "Curva de Retorno 180°": 2.2, "Tê (Fluxo Direto)": 0.6, "Tê (Fluxo Lateral)": 1.8,
}
FLUIDOS = { "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6}, "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6} }

# --- Funções de Callback ---
def adicionar_item(tipo_lista):
    novo_id = time.time()
    st.session_state[tipo_lista].append({"id": novo_id, "comprimento": 10.0, "diametro": 100.0, "material": "Aço Carbono (novo)", "acessorios": []})

def remover_ultimo_item(tipo_lista):
    if len(st.session_state[tipo_lista]) > 0: st.session_state[tipo_lista].pop()

def adicionar_ramal_paralelo():
    novo_nome_ramal = f"Ramal {len(st.session_state.ramais_paralelos) + 1}"
    novo_id = time.time()
    st.session_state.ramais_paralelos[novo_nome_ramal] = [{"id": novo_id, "comprimento": 50.0, "diametro": 80.0, "material": "Aço Carbono (novo)", "acessorios": []}]

def remover_ultimo_ramal():
    if len(st.session_state.ramais_paralelos) > 1: st.session_state.ramais_paralelos.popitem()

def adicionar_acessorio(id_trecho, lista_trechos):
    nome_acessorio = st.session_state[f"selectbox_acessorio_{id_trecho}"]
    quantidade = st.session_state[f"quantidade_acessorio_{id_trecho}"]
    for trecho in lista_trechos:
        if trecho["id"] == id_trecho:
            trecho["acessorios"].append({"nome": nome_acessorio, "k": K_FACTORS[nome_acessorio], "quantidade": int(quantidade)})
            break

# --- Funções de Cálculo ---
def calcular_perda_serie(lista_trechos, vazao_m3h, fluido_selecionado):
    perda_total = 0
    for trecho in lista_trechos:
        perdas = calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado)
        perda_total += perdas["principal"] + perdas["localizada"]
    return perda_total

def calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado):
    if vazao_m3h < 0: vazao_m3h = 0
    rugosidade_mm = MATERIAIS[trecho["material"]]
    vazao_m3s, diametro_m = vazao_m3h / 3600, trecho["diametro"] / 1000
    nu = FLUIDOS[fluido_selecionado]["nu"]
    if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
    area = (math.pi * diametro_m**2) / 4
    velocidade = vazao_m3s / area
    reynolds = (velocidade * diametro_m) / nu if nu > 0 else 0
    fator_atrito = 0
    if reynolds > 4000:
        rugosidade_m = rugosidade_mm / 1000
        log_term = math.log10((rugosidade_m / (3.7 * diametro_m)) + (5.74 / reynolds**0.9))
        fator_atrito = 0.25 / (log_term**2)
    elif reynolds > 0: fator_atrito = 64 / reynolds
    perda_principal = fator_atrito * (trecho["comprimento"] / diametro_m) * (velocidade**2 / (2 * 9.81))
    k_total_trecho = sum(ac["k"] * ac["quantidade"] for ac in trecho["acessorios"])
    perda_localizada = k_total_trecho * (velocidade**2 / (2 * 9.81))
    return {"principal": perda_principal, "localizada": perda_localizada, "velocidade": velocidade}

def calcular_perdas_paralelo(ramais, vazao_total_m3h, fluido_selecionado):
    num_ramais = len(ramais)
    if num_ramais < 2: return 0, {}
    lista_ramais = list(ramais.values())
    def equacoes_perda(vazoes_parciais_m3h):
        vazao_ultimo_ramal = vazao_total_m3h - sum(vazoes_parciais_m3h)
        if vazao_ultimo_ramal < -0.01: return [1e12] * (num_ramais - 1)
        todas_vazoes = np.append(vazoes_parciais_m3h, vazao_ultimo_ramal)
        perdas = [calcular_perda_serie(ramal, vazao, fluido_selecionado) for ramal, vazao in zip(lista_ramais, todas_vazoes)]
        erros = [perdas[i] - perdas[-1] for i in range(num_ramais - 1)]
        return erros
    chute_inicial = np.full(num_ramais - 1, vazao_total_m3h / num_ramais)
    solucao = root(equacoes_perda, chute_inicial, method='hybr')
    if not solucao.success: return -1, {}
    vazoes_finais = np.append(solucao.x, vazao_total_m3h - sum(solucao.x))
    perda_final_paralelo = calcular_perda_serie(lista_ramais[0], vazoes_finais[0], fluido_selecionado)
    distribuicao_vazao = {nome_ramal: vazao for nome_ramal, vazao in zip(ramais.keys(), vazoes_finais)}
    return perda_final_paralelo, distribuicao_vazao

def calcular_analise_energetica(vazao_m3h, h_man, eficiencia_bomba, eficiencia_motor, horas_dia, custo_kwh, fluido_selecionado):
    rho = FLUIDOS[fluido_selecionado]["rho"]
    potencia_eletrica_kW = (vazao_m3h / 3600 * rho * 9.81 * h_man) / (eficiencia_bomba * eficiencia_motor) / 1000 if eficiencia_bomba * eficiencia_motor > 0 else 0
    custo_anual = potencia_eletrica_kW * horas_dia * 30 * 12 * custo_kwh
    return {"potencia_eletrica_kW": potencia_eletrica_kW, "custo_anual": custo_anual}

def gerar_grafico_sensibilidade_diametro(sistema_base, fator_escala_range, **params_fixos):
    custos, fatores = [], np.arange(fator_escala_range[0], fator_escala_range[1] + 1, 5)
    for fator in fatores:
        escala = fator / 100.0
        sistema_escalado = {'antes': [t.copy() for t in sistema_base['antes']], 'paralelo': {k: [t.copy() for t in v] for k, v in sistema_base['paralelo'].items()}, 'depois': [t.copy() for t in sistema_base['depois']]}
        for t_list in sistema_escalado.values():
            if isinstance(t_list, list):
                for t in t_list: t['diametro'] *= escala
            elif isinstance(t_list, dict):
                for _, ramal in t_list.items():
                    for t in ramal: t['diametro'] *= escala
        perda_antes = calcular_perda_serie(sistema_escalado['antes'], params_fixos['vazao'], params_fixos['fluido'])
        perda_par, _ = calcular_perdas_paralelo(sistema_escalado['paralelo'], params_fixos['vazao'], params_fixos['fluido'])
        perda_depois = calcular_perda_serie(sistema_escalado['depois'], params_fixos['vazao'], params_fixos['fluido'])
        if perda_par == -1: custos.append(np.nan); continue
        h_man = params_fixos['h_geo'] + perda_antes + perda_par + perda_depois
        resultado_energia = calcular_analise_energetica(params_fixos['vazao'], h_man, **params_fixos['equipamentos'])
        custos.append(resultado_energia['custo_anual'])
    return pd.DataFrame({'Fator de Escala nos Diâmetros (%)': fatores, 'Custo Anual de Energia (R$)': custos})

def gerar_diagrama_rede(sistema, vazao_total, distribuicao_vazao, fluido):
    dot = graphviz.Digraph(comment='Rede de Tubulação'); dot.attr('graph', rankdir='LR', splines='ortho'); dot.attr('node', shape='point'); 
    dot.node('start', 'Bomba', shape='circle', style='filled', fillcolor='#ADD8E6', fontcolor='black', fontsize='10')
    dot.node('end', 'Fim', shape='circle', style='filled', fillcolor='#D3D3D3', fontcolor='black', fontsize='10')
    dot.attr('edge', color=PRIMARY_COLOR.hexval, fontcolor=PRIMARY_COLOR.hexval, fontsize='8')
    ultimo_no = 'start'
    for i, trecho in enumerate(sistema['antes']):
        proximo_no = f"no_antes_{i+1}"; velocidade = calcular_perdas_trecho(trecho, vazao_total, fluido)['velocidade']; label = f"Trecho Antes {i+1}\\nVazão: {vazao_total:.1f} m³/h\\nVelocidade: {velocidade:.2f} m/s"; dot.edge(ultimo_no, proximo_no, label=label); ultimo_no = proximo_no
    if len(sistema['paralelo']) >= 2 and distribuicao_vazao:
        no_divisao = ultimo_no; no_juncao = 'no_juncao'; dot.node(no_juncao)
        for nome_ramal, trechos_ramal in sistema['paralelo'].items():
            vazao_ramal = distribuicao_vazao.get(nome_ramal, 0); ultimo_no_ramal = no_divisao
            for i, trecho in enumerate(trechos_ramal):
                velocidade = calcular_perdas_trecho(trecho, vazao_ramal, fluido)['velocidade']; label_ramal = f"{nome_ramal} (T{i+1})\\nVazão: {vazao_ramal:.1f} m³/h\\nVelocidade: {velocidade:.2f} m/s"
                if i == len(trechos_ramal) - 1: dot.edge(ultimo_no_ramal, no_juncao, label=label_ramal)
                else: proximo_no_ramal = f"no_{nome_ramal}_{i+1}".replace(" ", "_"); dot.edge(ultimo_no_ramal, proximo_no_ramal, label=label_ramal); ultimo_no_ramal = proximo_no_ramal
        ultimo_no = no_juncao
    for i, trecho in enumerate(sistema['depois']):
        proximo_no = f"no_depois_{i+1}"; velocidade = calcular_perdas_trecho(trecho, vazao_total, fluido)['velocidade']; label = f"Trecho Depois {i+1}\\nVazão: {vazao_total:.1f} m³/h\\nVelocidade: {velocidade:.2f} m/s"; dot.edge(ultimo_no, proximo_no, label=label); ultimo_no = proximo_no
    dot.edge(ultimo_no, 'end')
    return dot

# --- FUNÇÃO PARA GERAR O RELATÓRIO PDF (NOVA) ---
def gerar_pdf_relatorio(resultados_analise, sistema_atual, vazao, distribuicao_vazao, fluido_selecionado, chart_data_sensibilidade):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Estilos personalizados
    styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=24, alignment=1, spaceAfter=20, textColor=PRIMARY_COLOR))
    styles.add(ParagraphStyle(name='Heading1', fontName='Helvetica-Bold', fontSize=18, spaceBefore=20, spaceAfter=10, textColor=PRIMARY_COLOR))
    styles.add(ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=10, leading=14))

    story = []

    # Título e Data
    story.append(Paragraph("Relatório de Análise de Rede Hidráulica", styles['TitleStyle']))
    story.append(Spacer(0, 0.5*cm))
    story.append(Paragraph(f"Data da Análise: {time.strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(0, 1*cm))

    # Resultados Principais
    story.append(Paragraph("1. Resumo da Análise (Caso Base)", styles['Heading1']))
    resumo_texto = f"""
        <b>Altura Manométrica Total:</b> {resultados_analise['h_man_total']:.2f} m<br/>
        <b>Perda de Carga Total:</b> {resultados_analise['perda_total_sistema']:.2f} m<br/>
        <b>Potência Elétrica Consumida:</b> {resultados_analise['potencia_eletrica_kW']:.2f} kW<br/>
        <b>Custo Anual de Energia:</b> R$ {resultados_analise['custo_anual']:.2f}
    """
    story.append(Paragraph(resumo_texto, styles['Normal']))
    story.append(Spacer(0, 1*cm))

    # Diagrama da Rede
    story.append(Paragraph("2. Diagrama da Rede", styles['Heading1']))
    if sistema_atual['paralelo'] and distribuicao_vazao:
        try:
            diagram_bytes = gerar_diagrama_rede(sistema_atual, vazao, distribuicao_vazao, fluido_selecionado).pipe(format='png')
            diagram_img_path = io.BytesIO(diagram_bytes)
            story.append(Image(diagram_img_path, width=18*cm, height=10*cm, kind='proportional'))
        except Exception as e:
            story.append(Paragraph(f"Não foi possível gerar o diagrama: {str(e)}", styles['Normal']))
    else:
        story.append(Paragraph("Diagrama não disponível para a configuração atual.", styles['Normal']))
    
    story.append(PageBreak())

    # Gráfico de Sensibilidade
    story.append(Paragraph("3. Análise de Sensibilidade de Custo", styles['Heading1']))
    if not chart_data_sensibilidade.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(chart_data_sensibilidade['Fator de Escala nos Diâmetros (%)'], chart_data_sensibilidade['Custo Anual de Energia (R$)'], color=PRIMARY_COLOR.hexval, marker='o')
        ax.set_title('Custo Anual vs. Fator de Escala do Diâmetro', color=PRIMARY_COLOR.hexval)
        ax.set_xlabel('Fator de Escala nos Diâmetros (%)', color=PRIMARY_COLOR.hexval)
        ax.set_ylabel('Custo Anual de Energia (R$)', color=PRIMARY_COLOR.hexval)
        ax.tick_params(colors=PRIMARY_COLOR.hexval)
        ax.grid(True, linestyle='--', alpha=0.6)
        chart_img_path = io.BytesIO()
        plt.tight_layout()
        plt.savefig(chart_img_path, format='png', dpi=300)
        plt.close(fig)
        chart_img_path.seek(0)
        story.append(Image(chart_img_path, width=16*cm))
    else:
        story.append(Paragraph("Gráfico de sensibilidade não foi gerado.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Inicialização do Estado da Sessão ---
if 'trechos_antes' not in st.session_state: st.session_state.trechos_antes = []
if 'trechos_depois' not in st.session_state: st.session_state.trechos_depois = []
if 'ramais_paralelos' not in st.session_state:
    st.session_state.ramais_paralelos = {
        "Ramal 1": [{"id": time.time(), "comprimento": 50.0, "diametro": 80.0, "material": "Aço Carbono (novo)", "acessorios": []}],
        "Ramal 2": [{"id": time.time() + 1, "comprimento": 50.0, "diametro": 100.0, "material": "Aço Carbono (novo)", "acessorios": []}]
    }

# --- Interface do Aplicativo ---
st.set_page_config(layout="wide", page_title="Análise de Redes Hidráulicas")
st.title("💧 Análise de Redes de Bombeamento (Série e Paralelo)")

def render_trecho_ui(trecho, prefixo, lista_trechos):
    st.markdown(f"**Trecho**"); c1, c2, c3 = st.columns(3)
    trecho['comprimento'] = c1.number_input("L (m)", min_value=0.1, value=trecho['comprimento'], key=f"comp_{prefixo}_{trecho['id']}")
    trecho['diametro'] = c2.number_input("Ø (mm)", min_value=1.0, value=trecho['diametro'], key=f"diam_{prefixo}_{trecho['id']}")
    trecho['material'] = c3.selectbox("Material", options=list(MATERIAIS.keys()), index=list(MATERIAIS.keys()).index(trecho.get('material', 'Aço Carbono (novo)')), key=f"mat_{prefixo}_{trecho['id']}")
    st.markdown("**Acessórios (Fittings)**")
    for idx, acessorio in enumerate(trecho['acessorios']): 
        col1, col2 = st.columns([0.8, 0.2])
        col1.info(f"{acessorio['quantidade']}x {acessorio['nome']} (K = {acessorio['k']})")
        if col2.button("X", key=f"rem_acc_{trecho['id']}_{idx}", help="Remover acessório"):
            trecho['acessorios'].pop(idx)
            st.rerun()
    c1, c2 = st.columns([3, 1]); c1.selectbox("Selecionar Acessório", options=list(K_FACTORS.keys()), key=f"selectbox_acessorio_{trecho['id']}"); c2.number_input("Qtd", min_value=1, value=1, step=1, key=f"quantidade_acessorio_{trecho['id']}")
    st.button("Adicionar Acessório", on_click=adicionar_acessorio, args=(trecho['id'], lista_trechos), key=f"btn_add_acessorio_{trecho['id']}", use_container_width=True)

with st.sidebar:
    st.header("⚙️ Parâmetros Gerais"); 
    st.session_state.fluido_selecionado = st.selectbox("Selecione o Fluido", list(FLUIDOS.keys()))
    st.session_state.vazao = st.number_input("Vazão Total (m³/h)", 0.1, value=100.0, step=1.0)
    st.session_state.h_geometrica = st.number_input("Altura Geométrica (m)", 0.0, value=15.0)
    st.divider()
    with st.expander("1. Trechos em Série (Antes da Divisão)"):
        for i, trecho in enumerate(st.session_state.trechos_antes):
            with st.container(border=True): render_trecho_ui(trecho, f"antes_{i}", st.session_state.trechos_antes)
        c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Antes)", on_click=adicionar_item, args=("trechos_antes",), use_container_width=True); c2.button("Remover Trecho (Antes)", on_click=remover_ultimo_item, args=("trechos_antes",), use_container_width=True)
    with st.expander("2. Ramais em Paralelo", expanded=True):
        for nome_ramal, trechos_ramal in st.session_state.ramais_paralelos.items():
            with st.container(border=True):
                st.subheader(f"{nome_ramal}")
                for i, trecho in enumerate(trechos_ramal): render_trecho_ui(trecho, f"par_{nome_ramal}_{i}", trechos_ramal)
        c1, c2 = st.columns(2); c1.button("Adicionar Ramal Paralelo", on_click=adicionar_ramal_paralelo, use_container_width=True); c2.button("Remover Último Ramal", on_click=remover_ultimo_ramal, use_container_width=True, disabled=len(st.session_state.ramais_paralelos) < 2)
    with st.expander("3. Trechos em Série (Depois da Junção)"):
        for i, trecho in enumerate(st.session_state.trechos_depois):
            with st.container(border=True): render_trecho_ui(trecho, f"depois_{i}", st.session_state.trechos_depois)
        c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Depois)", on_click=adicionar_item, args=("trechos_depois",), use_container_width=True); c2.button("Remover Trecho (Depois)", on_click=remover_ultimo_item, args=("trechos_depois",), use_container_width=True)
    st.divider(); st.header("🔌 Equipamentos e Custo")
    st.session_state.rend_bomba = st.slider("Eficiência da Bomba (%)", 1, 100, 70); st.session_state.rend_motor = st.slider("Eficiência do Motor (%)", 1, 100, 90)
    st.session_state.horas_por_dia = st.number_input("Horas por Dia", 1.0, 24.0, 8.0, 0.5); st.session_state.tarifa_energia = st.number_input("Custo da Energia (R$/kWh)", 0.10, 5.00, 0.75, 0.01, format="%.2f")

# --- Lógica Principal e Exibição de Resultados ---
try:
    # Coleta de valores do session_state
    vazao_calc = st.session_state.vazao; h_geometrica_calc = st.session_state.h_geometrica; rend_bomba_calc = st.session_state.rend_bomba
    rend_motor_calc = st.session_state.rend_motor; horas_por_dia_calc = st.session_state.horas_por_dia; tarifa_energia_calc = st.session_state.tarifa_energia
    fluido_selecionado_calc = st.session_state.fluido_selecionado
    
    # Cálculos
    perda_serie_antes = calcular_perda_serie(st.session_state.trechos_antes, vazao_calc, fluido_selecionado_calc)
    perda_paralelo, distribuicao_vazao = calcular_perdas_paralelo(st.session_state.ramais_paralelos, vazao_calc, fluido_selecionado_calc)
    perda_serie_depois = calcular_perda_serie(st.session_state.trechos_depois, vazao_calc, fluido_selecionado_calc)
    if perda_paralelo == -1:
        st.error("O cálculo do sistema em paralelo falhou. Verifique se os parâmetros dos ramais são consistentes."); st.stop()
    perda_total_sistema = perda_serie_antes + perda_paralelo + perda_serie_depois
    h_man_total = h_geometrica_calc + perda_total_sistema
    params_equipamentos = {'eficiencia_bomba': rend_bomba_calc/100, 'eficiencia_motor': rend_motor_calc/100, 'horas_dia': horas_por_dia_calc, 'custo_kwh': tarifa_energia_calc, 'fluido_selecionado': fluido_selecionado_calc}
    resultados_energia = calcular_analise_energetica(vazao_calc, h_man_total, **params_equipamentos)

    st.header("📊 Resultados da Análise da Rede (Caso Base)"); 
    c1,c2,c3,c4 = st.columns(4); c1.metric("Altura Total", f"{h_man_total:.2f} m"); c2.metric("Perda Total", f"{perda_total_sistema:.2f} m"); c3.metric("Potência Elétrica", f"{resultados_energia['potencia_eletrica_kW']:.2f} kW"); c4.metric("Custo Anual", f"R$ {resultados_energia['custo_anual']:.2f}")
    
    st.header("🗺️ Diagrama da Rede")
    sistema_atual = {'antes': st.session_state.trechos_antes, 'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
    diagram = None
    if distribuicao_vazao:
        diagram = gerar_diagrama_rede(sistema_atual, vazao_calc, distribuicao_vazao, fluido_selecionado_calc)
        st.graphviz_chart(diagram)
    else: st.info("O diagrama será exibido quando houver um cálculo paralelo bem-sucedido.")
    
    st.divider()
    chart_data_sensibilidade = pd.DataFrame()
    with st.expander("📈 Análise de Sensibilidade de Custo por Diâmetro"):
        escala_range = st.slider("Fator de Escala para Diâmetros (%)", 50, 200, (80, 120))
        params_fixos = {'vazao': vazao_calc, 'h_geo': h_geometrica_calc, 'fluido': fluido_selecionado_calc, 'equipamentos': params_equipamentos}
        chart_data_sensibilidade = gerar_grafico_sensibilidade_diametro(sistema_atual, escala_range, **params_fixos)
        if not chart_data_sensibilidade.empty:
            st.line_chart(chart_data_sensibilidade.set_index('Fator de Escala nos Diâmetros (%)'))
    
    st.divider()
    st.header("📄 Gerar Relatório Executivo")
    st.markdown("Clique no botão abaixo para gerar um relatório completo em PDF com todos os resultados, diagrama e gráfico de análise.")
    resultados_para_pdf = {'h_man_total': h_man_total, 'perda_total_sistema': perda_total_sistema, 'potencia_eletrica_kW': resultados_energia['potencia_eletrica_kW'], 'custo_anual': resultados_energia['custo_anual']}
    
    # Gerar o PDF em memória quando o botão for clicado
    pdf_buffer = gerar_pdf_relatorio(
        resultados_para_pdf, 
        sistema_atual, 
        vazao_calc, 
        distribuicao_vazao, 
        fluido_selecionado_calc, 
        chart_data_sensibilidade
    )
    
    st.download_button(
        label="Download do Relatório em PDF",
        data=pdf_buffer,
        file_name=f"Relatorio_Hidraulico_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

except Exception as e:
    st.error(f"Ocorreu um erro durante o cálculo. Verifique os parâmetros de entrada. Detalhe: {str(e)}")
