import csv


import matplotlib.pyplot as plt
from datetime import datetime
import time
import shelve
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
import math
import plotly.plotly as py
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import linear_model
from random import uniform, randrange


# Classe para manipulação e inserção de dados na estrutura
class estruturaDados:

    # Leitura do arquivo padrão
    def executaLeitura(arquivo):
        estrutura = pd.read_csv(arquivo)
        data = []
        hora = []
        datahora = []
        for dado in estrutura['datahora']:
            data.append(datetime.strptime(str(dado), "%Y%m%d%H%M").date())
            hora.append(datetime.strptime(str(dado), "%Y%m%d%H%M").time())
            datahora.append(datetime.strptime(str(dado), "%Y%m%d%H%M"))

        estrutura["data"] = data
        estrutura["hora"] = hora
        estrutura["datahora"] = datahora



        return estrutura


    # Retorna a estrutura Dados somente com os ativos desejados (codigo)
    def estruturaDadosAtivo(estrutura, codigo):
        estruturaRetorno = []
        for t in estrutura:
            for k in codigo:
                if t.codigo == k:
                    estruturaRetorno.append(t)

        return estruturaRetorno

    # Retorna a Estrutura Dados para as datas específicas
    def periodoDados(estrutura, inicio, fim):
        return estrutura[(estrutura["data"]>datetime.strptime(str(inicio), "%d/%m/%Y").date()) & (estrutura["data"]<datetime.strptime(str(fim), "%d/%m/%Y").date())]

    def plot(estrutura,codigo):
        legenda = []
        data = []
        for i in codigo:
            if not estrutura[estrutura["codigo"] == i].empty:
                dado1 = estrutura[estrutura["codigo"] == i].fechamento_atual.values
                dado2 = estrutura[estrutura["codigo"] == i].datahora.values
                plotar =pd.Series(dado1, dado2)
                plotar.plot()
                legenda.append(i)

        plt.legend(legenda)
        plt.title("Histórico")
        plt.show()

        #data = data.append(pd.Series(estrutura[estrutura["codigo"] == i].fechamento_atual.values,estrutura[estrutura["codigo"] == i].data.values))
        #estrutura[estrutura["codigo"] == i].fechamento_atual.plot()


    def plotHist(estrutura,codigo):
        legenda = []
        for i in codigo:
            if not estrutura[estrutura["codigo"] == i].empty:
                sns.distplot(estrutura[estrutura["codigo"] == i].fechamento_atual, hist=True, kde=True,
                             bins=20).set_title("Histogramas")
                legenda.append(i)

        plt.legend(legenda)

    def plotHist_Ativo(estrutura,codigo):
        for i in codigo:
            if not estrutura[estrutura["codigo"] == i].empty:
                plt.figure()
                sns.distplot(estrutura[estrutura["codigo"] == i].fechamento_atual, hist=True, kde=True,
                             bins=20).set_title("Histograma " + i)

    def estatisticaDescritiva(estrutura,codigo):
        retorno = []
        for i in codigo:
            if not estrutura[estrutura["codigo"] == i].empty:
                aux = []
                aux.append(i)
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.mean())
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.std())
                aux.append((estrutura[estrutura["codigo"] == i].fechamento_atual.std() / estrutura[
                    estrutura["codigo"] == i].fechamento_atual.mean()) * 100)
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.min())
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.max())
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.quantile(q=0.25))
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.median())
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.quantile(q=0.75))

                retorno.append(aux)

        return pd.DataFrame(retorno, columns=['Ativo', 'Media', 'DesvP', 'Prop%_MedDesvP', 'Minimo', 'Máximo', '25%',
                                              'Mediana', '75%']).T

    def correlacao(estrutura,codigo):
        colunas = ['datahora']
        for i in codigo:
            colunas.append(i)
        saida = pd.DataFrame([],columns= colunas)
        data = estrutura[estrutura["codigo"] == codigo[0]].datahora
        saida[colunas[0]] = data
        #saida[colunas[1]] =estrutura[estrutura["codigo"] == codigo[0]].fechamento_atual
        for j in codigo:
            if not estrutura[estrutura["codigo"] == j].empty:
                dados = []
                for i in data:
                    dados.append(estrutura[(estrutura["codigo"] == j) & (estrutura["datahora"] == i)].fechamento_atual)
                saida[j] = dados
        print(saida)

        return pd.DataFrame(saida)


    def boxplot(estrutura,codigo):
        aux = []
        legenda = []
        for i in codigo:
            if not estrutura[estrutura["codigo"] == i].empty:
                aux.append(estrutura[estrutura["codigo"] == i].fechamento_atual.values)
                legenda.append(i)



        plt.boxplot(aux, labels = legenda)


    def save_variables(globals_=None):
        if globals_ is None:
            globals_ = globals()
        filename = 'dados.out'
        my_shelf = shelve.open(filename, 'n')
        for key, value in globals_.items():
            #if not key.startswith('__'):
            if key == "dadosCompleto":
                try:
                        my_shelf[key] = value
                except Exception:
                    print('ERROR shelving: "%s"' % key)
        my_shelf.close()

    def loadDados(arquivo):
        my_shelf = shelve.open(arquivo)
        for key in my_shelf:
            globals()[key] = my_shelf[key]
        my_shelf.close()


    def MontaSerie_Cluster(estrutura,codigo):
        estrutura = estrutura.dropna();
        colunas = ['datahora']
        for i in codigo:
            colunas.append(i)
        # print(colunas)
        saida = pd.DataFrame([], columns=colunas)
        data = estrutura[estrutura["codigo"] == codigo[0]].datahora
        saida[colunas[0]] = data
        for ativo in codigo:
            aux = []
            aux2 = []
            print("Asset: " + ativo)
            auxData = estrutura[(estrutura["codigo"] == ativo)].datahora.values
            aux = estrutura[(estrutura["codigo"] == ativo)].fechamento_atual.values
            if not estrutura[estrutura["codigo"] == ativo].empty:
                dados = []
                for dataBusca in data:
                    index = -1
                    i = 0
                    for estAuxData in auxData:
                        if (dataBusca == estAuxData):
                            index = i
                            break
                        i = i + 1
                    if index != -1:
                        aux2.append(aux[index])
                        auxData = np.delete(auxData, index)
                        aux = np.delete(aux, index)
                    else:
                        aux2.append(np.nan)
                saida[ativo] = aux2
        saida = saida.dropna();
        print("Successfully Executed!!!")
        return saida

    def clusteringHierarchial(estrutura, codigo):
        dadosCluster = estrutura[codigo].T


        Z = hac.linkage(dadosCluster, method='average', metric='correlation')

        # Plot dendogram
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        hac.dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.show()
        return Z

    def segmentarClusters_clusteringHierarchial(Z, k):
        listaCluster = []
        # k Number of clusters I'd like to extract
        results = fcluster(Z, k, criterion='maxclust')

        # check the results
        s = pd.Series(results)
        clusters = s.unique()
        for c in clusters:
            listaCluster.append((s[s == c]).index)
            cluster_indeces = s[s == c].index
            print("Cluster %d number of entries %d" % (c, len(cluster_indeces)))

        return listaCluster

    def implimirCluster(estrutura, estruturaSeries, listaCluster, identificador):
        lista = list(estruturaSeries.T.iloc[listaCluster[identificador]].index)
        estruturaDados.plot(estrutura, lista)

    def logRetorno(dadosSeries, ativosAnalisados):
        DadosLogRetorno = pd.DataFrame([])
        DadosLogRetorno["datahora"] = dadosSeries.datahora
        for ativo in ativosAnalisados:
            auxLogRetorno = []
            flag = True;
            for dado in dadosSeries[ativo]:
                if flag:
                    auxLogRetorno.append(0)
                    aux = dado
                    flag = False
                else:
                    auxLogRetorno.append(math.log(dado / aux))
                    aux = dado
            DadosLogRetorno[ativo] = auxLogRetorno
        return DadosLogRetorno

    def segmentaDias(DadosPeriodo, DadosEntrada, AtivoTeste):

        # Cluster por Dia
        dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]

        hora = sorted(DadosPeriodo.hora.unique())
        hora = hora[0:32]  # Somente dados durante o pregão... Exclui dados pós pregão
        DadosDiaSemana = pd.DataFrame([], columns=hora)
        DadosDiaSemana.insert(0, 'Semana', [], allow_duplicates=False)
        DadosDiaSemana.insert(0, 'Dia', [], allow_duplicates=False)

        data = DadosPeriodo.data.unique()
        i = 0

        for datas in data:
            # DadosDiaSemana.set_value(i, 'Dia', dados)
            DadosDiaSemana = DadosDiaSemana.append({'Dia': datas, 'Semana': dias[datas.weekday()]}, ignore_index=True)
            aux = 0
            for horas in hora:
                teste = str(datas) + '-' + str(horas)
                teste = datetime.strptime(str(teste), "%Y-%m-%d-%H:%M:%S")
                valorSerie = DadosEntrada[DadosEntrada["datahora"] == teste][AtivoTeste]
                if not valorSerie.empty:
                    DadosDiaSemana.loc[[i], [horas]] = valorSerie.values
                    aux = valorSerie.values
                else:
                    if i == 0:
                        DadosDiaSemana.loc[[i], [horas]] = 0
                    else:
                        DadosDiaSemana.loc[[i], [horas]] = aux
            i = i + 1

        i = 0
        for datas in data:
            a = (DadosDiaSemana.iloc[[i], [2]] == 0)
            if (a.values):
                if (DadosDiaSemana.iloc[[i], [3]].values == 0):
                    DadosDiaSemana.iloc[[i], [2]] = np.nan
                else:
                    DadosDiaSemana.iloc[[i], [2]] = DadosDiaSemana.iloc[[i], [3]].values
            i = i + 1

        return DadosDiaSemana

    # Retorna o Log Retorno
    def DadosEntrada(DadosTratados, hora, ponderador):

        # DadosTratados = DadosTratados.T

        logRetornoTratado = pd.DataFrame([], columns=hora[0:31])

        # logRetornoTratado = []
        for dia in range(0, len(DadosTratados)):
            logRetornoAux = []
            for tempo in range(0, len(DadosTratados.T)):
                if tempo == 0:
                    inicio = DadosTratados.iloc[dia][tempo]
                else:
                    logRetornoAux.append(ponderador * (1 + (math.log(DadosTratados.iloc[dia][tempo] / inicio))))
            logRetornoTratado.loc[dia] = logRetornoAux

        return logRetornoTratado.T

    # Retira Nan, os dados de Dia da Semana
    def tratamentoDados(DadosDiaSemana):
        DadosTratados = DadosDiaSemana.drop("Semana", axis=1)

        DadosTratados = DadosTratados.dropna()

        Temp = DadosTratados
        # Temp["Dia"].values
        DiasAnalisados = Temp["Dia"].values

        DadosTratados = DadosTratados.drop("Dia", axis=1)

        CorrigeIndex = DadosTratados
        indexTeste = range(0, len(CorrigeIndex))
        CorrigeIndex.index = indexTeste

        DadosTratados = CorrigeIndex

        return DadosTratados, DiasAnalisados[1:]

    # Verifica a tendencia das curvas dos clusters
    def tendencia(DadosTestando, curva):
        aux = 0;
        cont = 0;
        var = [];
        auxvar = 0;
        for variacao in curva:
            if aux != 0:
                if aux < variacao:
                    var.append(1)
                else:
                    if aux > variacao:
                        var.append(-1)
                    else:
                        var.append(0)
            aux = variacao

        flag = 0
        inclinacao = var[1]
        varAbs = DadosTestando['Media'].values;
        saida = []
        pos = []
        if inclinacao == 1:
            saida.append("Upward Trend")
        if (inclinacao == -1):
            saida.append("Downtrend")
        if (inclinacao == 0):
            saida.append("Later Trend")
        pos.append(1)
        cont = 0
        acumulado = 0
        index = 1
        for i in range(0, len(var)):
            if flag > 1:
                if (var[i] == var[index]):
                    cont = 0
                    acumulado = 0
                if (var[i] != var[index]):
                    cont = cont + 1
                    acumulado = acumulado + abs(varAbs[index])
                if cont > 2:
                    index = i
                    pos.append(index)
                    cont = 0;
                    if ((var[index] == -1) & (acumulado > 0.2)):
                        # print(" -- Tendência de Baixa")
                        saida.append("Downtrend")
                        # variacoes.append(var[index],'alta')
                    else:
                        if ((var[index] == 1) & (acumulado > 0.2)):
                            #    print(" -- Tendência de Alta")
                            saida.append("Upward Trend")
                            # variacoes.append(var[index],'baixa')
                        else:
                            #   print(" -- Tendência Lateral")
                            saida.append("Later Trend")
                            # variacoes.append(var[index],'lateral')

            else:
                flag = flag + 1

        return saida, pos

        # Retorna as médias dos clusters e as variações discretizadas na granularidade dos dados

    def CurvasClusters(DadosNorm, listaCluster, listaOrdenada):
        curvasClusters = []
        curvasMedias = []
        for i in range(0, len(listaCluster)):
            curvaIsolada = []
            curvaMedia = []
            lista = listaCluster[listaOrdenada[i]]
            DadosMediaCluster = pd.DataFrame(DadosNorm.iloc[lista].mean(), columns=['Media'])
            flag = False;
            horas = DadosMediaCluster.index
            for y in range(0, len(DadosMediaCluster.index)):
                curvaMedia.append(DadosMediaCluster['Media'][y])
                if flag:
                    if DadosMediaCluster['Media'][y] > DadosMediaCluster['Media'][y - 1]:
                        curvaIsolada.append(1);
                    else:
                        curvaIsolada.append(-1);
                else:
                    flag = True
            curvasClusters.append(curvaIsolada)
            curvasMedias.append(curvaMedia)

        return curvasClusters, curvasMedias
        # [tendencias, pos] = tendencia(list(DadosTestando['Media'].values))
        # DadosNorm[lista].T.mean()

    def DadosToCluster(logRetornoTratado, plot):
        Z = hac.linkage(logRetornoTratado, method='average', metric='correlation')
        # Z = hac.linkage(DadosTratados[inicio:fim], method='average', metric='correlation')
        if plot:
            # Plot dendogram
            plt.figure(figsize=(25, 10))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Index')
            plt.ylabel('Distance')
            hac.dendrogram(
                Z,
                leaf_rotation=60.,  # rotates the x axis labels
                leaf_font_size=8.,  # font size for the x axis labels
            )
            plt.show()
        return Z

    def segmentaCluster(Z, K, plot):
        # Lista a quantidade em cada cluster
        listaCluster = []
        # k Number of clusters I'd like to extract
        results = fcluster(Z, K, criterion='maxclust')

        # print("results: ", results)
        print("Quantity Days: ", len(results))
        # check the results
        s = pd.Series(results)
        clusters = s.unique()
        i = 0
        for c in clusters:
            listaCluster.append((s[s == c]).index)
            cluster_indices = s[s == c].index
            if (len(cluster_indices) >= 4) & (plot):
                print("Cluster %d have %d Weeks: Index[%d]" % (c, len(cluster_indices), i))
            i = i + 1

        return listaCluster

    def ordenacaoTamanhoCluster(DadosTratados, listaCluster):
        Ordenacao = []
        Representacao = []
        Soma = 0
        for lista in listaCluster:
            Ordenacao.append(len(lista))
            Soma = Soma + len(lista)

        listaOrdenada = sorted(range(len(Ordenacao)), key=Ordenacao.__getitem__, reverse=True)

        for i in range(0, len(listaCluster)):
            lista = listaCluster[listaOrdenada[i]]
            Percentual = (len(lista) / Soma) * 100
            Representacao.append(Percentual)

        return listaOrdenada, Soma, Representacao

    # Retorna as médias dos clusters e as variações discretizadas na granularidade dos dados
    def CurvasClusters2(DadosNorm, listaCluster, listaOrdenada):
        curvasClusters = []
        curvasMedias = []
        for i in range(0, len(listaCluster)):
            curvaIsolada = []
            curvaMedia = []
            lista = listaCluster[listaOrdenada[i]]
            DadosMediaCluster = pd.DataFrame(DadosNorm.iloc[lista].mean(), columns=['Media'])
            # plt.plot(DadosMediaCluster)
            # plt.show()
            flag = False;
            horas = DadosMediaCluster.index
            for y in range(0, len(DadosMediaCluster.index)):
                curvaMedia.append(DadosMediaCluster['Media'][y])
                if flag:
                    if DadosMediaCluster['Media'][y] > DadosMediaCluster['Media'][y - 1]:
                        curvaIsolada.append(1);
                    else:
                        curvaIsolada.append(-1);
                else:
                    flag = True
            curvasClusters.append(curvaIsolada)
            curvasMedias.append(curvaMedia)

        return curvasClusters, curvasMedias

    # Print Dados Clusterizados
    def printClusters(listaCluster, Soma, Representacao, listaOrdenada, DadosTreinamento):
        Acumulado = 0
        for i in range(0, len(listaCluster)):
            lista = listaCluster[listaOrdenada[i]]
            Percentual = (len(lista) / Soma) * 100
            Representacao.append(Percentual)
            if (Acumulado <= 80) & (len(lista) > 4):
                print('Cluster %d' % (i))
                print('Have %d days' % len(lista))
                Acumulado = Percentual + Acumulado
                print('Representation: %.2f' % Percentual)
                print('Accumulated Representation: %.2f' % Acumulado)

                # ------------------------------//------------------------------#
                # Inclinação do Cluster
                # DadosTestando = pd.DataFrame(DadosTreinamento.[lista].T.mean(), columns=['Media'])
                DadosTestando = pd.DataFrame(DadosTreinamento.iloc[lista].mean(), columns=['Media'])
                # carregando dados hipotéticos, para fins didáticos apenas
                dataframe = pd.DataFrame()
                dataframe['x'] = np.linspace(0, len(DadosTestando['Media'].values), len(DadosTestando['Media'].values),
                                             endpoint=False)  # Entrada X
                dataframe['y'] = DadosTestando['Media'].values  # Média dos Valores do Cluster
                x_values = dataframe[['x']]
                y_values = dataframe[['y']]

                # treinando o modelo
                model = linear_model.LinearRegression()
                model.fit(x_values, y_values)

                # ------------------------------//------------------------------#
                # Modelo Linear com Erro
                predito = model.predict(x_values)
                ErroAbsoluto = abs(y_values - predito)  # Erro absoluto
                print('Absolute Error: %.4f' % ErroAbsoluto.sum());
                ErroQuadrático = pow(ErroAbsoluto, 2)  # Erro Quadrático
                MSE = ErroAbsoluto / len(ErroAbsoluto)  # Erro Quadrático Médio
                print('Mean Square Error: %.4f' % MSE.sum())

                [tendencias, pos] = estruturaDados.tendencia(DadosTestando, DadosTestando['Media'].values)

                for j in range(0, len(tendencias)):
                    print(str(tendencias[j]), end=" ")
                    print(str(DadosTestando.index[pos[j]]))

                # ------------------------------//------------------------------#
                # Plot Cluter
                DadosAux = DadosTreinamento.iloc[lista]
                DadosPlot = DadosAux.T
                # DadosPlot.plot(legend=False, alpha = 0.5)
                # DadosTestando.plot(legend=True, color='green', marker='o', label='Média',  linewidth=3)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(DadosPlot, alpha=0.5)
                ax.plot(DadosTestando, color='green', marker='o', label='Average', linewidth=3)
                ax.legend()

                # DadosNorm.plot(y = lista, legend=False, alpha=0.5);
                # DadosNorm[lista].T.mean().plot(legend=True, color='green', marker='o', label='Média',  linewidth=3)
                plt.title('Cluster %d' % (i))
                plt.xlabel('Time')
                plt.ylabel('Normalized Return')
                plt.show()

                # ------------------------------//------------------------------#
                # Distribuição do Cluster
                listaDistribuicao = []
                for k in DadosTreinamento.iloc[lista]:
                    for j in DadosTreinamento.iloc[lista][k]:
                        listaDistribuicao.append(j)

                # sns.distplot(DadosNorm[lista].T.mean(), hist=True, kde=True, bins=20).set_title("Distribuição do Cluster")
                sns.distplot(listaDistribuicao, hist=True, kde=True, bins=20).set_title("Distribuição do Cluster")
                plt.xlabel('Error')
                plt.ylabel('Frequency')
                plt.show()
                # ------------------------------//------------------------------#
                # Plot Regressão
                # plt.plot(DadosTestando.index, predito, color='blue', linewidth=3)
                # plt.scatter(x = DadosTestando.index, y=DadosTestando['Media'], color='red')
                # plt.title('Regressão Linear')
                # plt.xlabel('Hora')
                # plt.ylabel('Retorno Normalizado')

                # plt.show()

                # ------------------------------//------------------------------#
                # Plot Regressão

                # sns.distplot(y_values - predito, hist=True, kde=False).set_title('Distribuição do Erro Absoluto');
                # plt.xlabel('Erro')
                # plt.ylabel('Frequencia')
                # plt.show()

                print('#------------------------------//------------------------------#')

    def EvolucaoDiferencial(nPop, taxaCruzamento, F, DadosTreinamento,listaMedias, Representacao, Dados):
        # nPop = 40;
        # taxaCruzamento = 70;
        # F = 1;

        MGanhoPadrao = 0.002;
        MGanhoCorrelacao = 0.8;
        MRiscoCorrelacao = 0.7;
        MRiscoPadrao = 0;

        pop = estruturaDados.criaPop(nPop, MGanhoPadrao, MGanhoCorrelacao, MRiscoCorrelacao, MRiscoPadrao)
        QtGeracoes = 5
        for i in range(0, QtGeracoes):
            print("--------//--------")
            print("Generation: ", i + 1)

            [newPop, AuxMelhorFitness, retornoMelhorFitness] = estruturaDados.operadoresGeneticos(pop, taxaCruzamento, F,
                                                                                   DadosTreinamento,listaMedias, Representacao, Dados)

            if i == 0:
                FitnessAtual = AuxMelhorFitness
                estruturaParametros = retornoMelhorFitness
            else:
                if AuxMelhorFitness > FitnessAtual:
                    FitnessAtual = AuxMelhorFitness
                    estruturaParametros = retornoMelhorFitness

            print(estruturaParametros)
            pop = newPop

        return estruturaParametros

    def criaPop(NPop, MGanhoPadrao, MGanhoCorrelacao, MRiscoCorrelacao, MRiscoPadrao):
        pop = []
        for i in range(NPop):
            pop.append([0] * 4)

        for i in range(NPop):
            for j in range(4):
                if (j == 0):
                    if (i == 0):
                        pop[i][j] = MGanhoPadrao
                    else:
                        pop[i][j] = MGanhoPadrao + uniform(-0.0001, 0.0001)
                if (j == 1):
                    if (i == 0):
                        pop[i][j] = MGanhoCorrelacao
                    else:
                        pop[i][j] = MGanhoCorrelacao + uniform(-0.005, 0.005)
                if (j == 2):
                    if (i == 0):
                        pop[i][j] = MRiscoCorrelacao
                    else:
                        pop[i][j] = MRiscoCorrelacao + uniform(-0.005, 0.005)
                if (j == 3):
                    if (i == 0):
                        pop[i][j] = MRiscoPadrao
                    else:
                        pop[i][j] = MRiscoPadrao + uniform(-0.0001, 0.0001)

        return pop

    def operadoresGeneticos(Pop, taxaCruzamento, F, DadosTreinamento, listaMedias, Representacao, Dados):
        newPop = []
        auxPop = []
        AuxMelhorFitness = -99999999999999999999;
        stoplossAux = 0.001;
        for i in range(len(Pop)):
            newPop.append([0] * 4)
            auxPop.append([0] * 4)

        SomaFit = 0;
        contFit = 0
        for i in range(len(Pop)):
            for j in range(4):
                # Mutação e Cruzamentos Juntos
                if uniform(0, 1) < taxaCruzamento / 10:
                    a1 = i
                    # a1 = randrange(0, len(Pop))
                    # while(a1 == i):
                    # a1 = randrange(0, len(Pop))
                    a2 = randrange(0, len(Pop))
                    while ((a2 == i) | (a2 == a1)):
                        a2 = randrange(0, len(Pop))
                    a3 = randrange(0, len(Pop))
                    while ((a3 == i) | (a3 == a1) | (a3 == a2)):
                        a3 = randrange(0, len(Pop))

                    auxPop[i][j] = Pop[a1][j] + F * (Pop[a2][j] - Pop[a3][j])
                else:
                    auxPop[i][j] = Pop[a1][j]

            # Fitness para cada Parâmetro
            lista = []
            lista.append(auxPop[i][0])
            lista.append(auxPop[i][1])
            lista.append(auxPop[i][2])
            lista.append(auxPop[i][3])
            parametrosNegociacaoModificador = pd.DataFrame([lista], columns=['MGanhoPadrao', 'MGanhoCorrelacao',
                                                                             'MRiscoCorrelacao', 'MRiscoPadrao'],
                                                           index=[0])

            Fitness1 = 0


            for dia in range(len(DadosTreinamento) - 24, len(DadosTreinamento)):
                ganho = estruturaDados.SimulacaoEstrategia(dia, False, stoplossAux, DadosTreinamento, listaMedias, Representacao, parametrosNegociacaoModificador, Dados)
                Fitness1 = Fitness1 + ganho[0]
            # print("Vetor Modificador:", Fitness1)

            # Fitness para cada Parâmetro
            lista = []
            lista.append(Pop[i][0])
            lista.append(Pop[i][1])
            lista.append(Pop[i][2])
            lista.append(Pop[i][3])
            parametrosNegociacaoAlvo = pd.DataFrame([lista],
                                                    columns=['MGanhoPadrao', 'MGanhoCorrelacao', 'MRiscoCorrelacao',
                                                             'MRiscoPadrao'], index=[0])

            Fitness2 = 0
            for dia in range(len(DadosTreinamento) - 24, len(DadosTreinamento)):

                ganho = estruturaDados.SimulacaoEstrategia(dia, False, stoplossAux, DadosTreinamento, listaMedias, Representacao, parametrosNegociacaoAlvo, Dados)
                Fitness2 = Fitness2 + ganho[0]
            # print("Vetor Alvo:", Fitness2)

            if Fitness2 >= Fitness1:
                # print(parametrosNegociacaoModificador)
                SomaFit = SomaFit + Fitness2
                contFit = contFit + 1
                for k in range(4):
                    newPop[i][k] = Pop[i][k]
                if Fitness2 > AuxMelhorFitness:
                    AuxMelhorFitness = Fitness2
                    retornoMelhorFitness = parametrosNegociacaoModificador
            else:
                # print(parametrosNegociacaoAlvo)
                SomaFit = SomaFit + Fitness1
                contFit = contFit + 1
                for k in range(4):
                    newPop[i][k] = auxPop[i][k]

                if Fitness1 > AuxMelhorFitness:
                    AuxMelhorFitness = Fitness1
                    retornoMelhorFitness = parametrosNegociacaoAlvo

            # print("Interação: ", contFit)

        print("Media Geração: ", (SomaFit / contFit))
        return newPop, AuxMelhorFitness, retornoMelhorFitness

    # Calcula o Retorno por dia
    def SimulacaoEstrategia(dia, plot, stoplossAux, DadosEntrada, listaMedias, Representacao, parametrosNegociacao, Dados):

        hora = sorted(Dados.hora.unique())
        hora = hora[0:31]

        if (plot):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(DadosEntrada.iloc[dia])

        ## Estrutura Correlação
        cabecalho = ['curva']

        # Teste Apenas os 7 primeiros clusters
        cabecalho.extend([*range(0, 7)])

        # cabecalho.extend(list(range(0,len(listaMedias)))) ---- TODOS OS CLUSTERS
        correlacaoMomento = pd.DataFrame(columns=cabecalho);
        ##

        ultimoNegocio = 30  # Indice limite para negociacao
        valorCompraVenda = DadosEntrada.iloc[dia][ultimoNegocio]
        inicio = DadosEntrada.iloc[dia][0]
        stopLossCoberto = inicio * 2
        stopLossDescoberto = 0

        posCorrelacao = 0

        controleCompra = 0  # Controla de quanto em quanto tempo irá efetuar uma nova compra

        plotVerificaCompra = False  # Esse plot imprime variáveis enquanto roda o Modelo

        ## Controlam a quantidade de ativos vendidos e comprados
        compras = 0
        vendas = 0

        op = 0  # Controla a quantidade de Operações
        ##

        ganhoTotal = 0
        CompradoHoraCompra = 0
        CompradoHoraVenda = 0
        VendidoHoraCompra = 0
        VendidoHoraVenda = 0
        if inicio == 0:
            print("Day With Data Problem", dia)
        else:
            for i in range(0, len(DadosEntrada.iloc[dia]) - 1):
                correlacaoPontos = []
                correlacaoPontos.append(1 + (math.log(DadosEntrada.iloc[dia][i] / inicio)))
                # Apensa os 7 primeiros clusters
                # for j in range(0,len(listaMedias)): #---- TODOS OS CLUSTERS
                for j in range(0, 7):
                    correlacaoPontos.append(listaMedias[j][i - 2])

                correlacaoMomento.loc[posCorrelacao] = correlacaoPontos
                if i > 6:
                    Correlacao = correlacaoMomento.corr('pearson')
                    # print(Correlacao['curva'][1:])
                    if plot:
                        print("Hora: ", hora[i])

                    if (i < ultimoNegocio) & (controleCompra >= 0):
                        Negocio = estruturaDados.verificaCompra(Representacao, Correlacao['curva'][1:], listaMedias, i, ultimoNegocio,
                                                 plotVerificaCompra, compras, vendas, parametrosNegociacao);
                        if Negocio == 1:
                            if compras == 0:
                                # Simulação de Compra

                                valorAtivo = DadosEntrada.iloc[dia][i]
                                if (valorAtivo != 0):
                                    # [quantidade, totalCapital] = simulaCompra(totalCapital, valorAtivo, lote, custoOperacao, emolumentosTaxaDayTrade)
                                    valorCompra = valorAtivo
                                    stopLossCoberto = valorCompra * (1 - stoplossAux)
                                    op = op + 1;
                                    compras = compras + 1;
                                    CompradoHoraCompra = hora[i]

                                    HoraCompra = hora[i]

                                if (plot):
                                    ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='b', marker="^")
                                    print('Buy: %.2f' % valorAtivo)

                        if Negocio == -1:
                            if compras > 0:
                                # Simulação de Venda
                                ganhoTotal = ganhoTotal + (DadosEntrada.iloc[dia][i] - valorCompra)
                                # retorno = retorno + (DadosDiaSemana.iloc[dia][i] - Valorcompra);
                                compras = compras - 1;
                                HoraVenda = hora[i]
                                CompradoHoraVenda = hora[i]
                                if (plot):
                                    print('Sold: %.2f' % DadosEntrada.iloc[dia][i])
                                    ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='r', marker="v")

                                # print('venda: %.2f' %DadosDiaSemana.iloc[dia][i])
                                # print('retorno: %.2f' %retorno)
                                valorCompraVenda = DadosEntrada.iloc[dia][i]

                                controleCompra = -1
                        # StopLoss Comprado
                        if (DadosEntrada.iloc[dia][i] < stopLossCoberto):
                            if compras > 0:
                                valorAtivo = DadosEntrada.iloc[dia][i]
                                ganhoTotal = ganhoTotal + (int(stopLossCoberto) - valorCompra)
                                CompradoHoraVenda = hora[i]

                                HoraVenda = hora[i]

                                if (plot):
                                    ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='k', marker="v")
                                    print('Sold: %.2f' % DadosEntrada.iloc[dia][i])
                                compras = compras - 1;
                                valorCompraVenda = DadosEntrada.iloc[dia][i]
                                controleCompra = -1

                        # --------------------//--------------------
                        # Venda / Compra
                        if Negocio == 2:  # Venda Descoberda
                            if vendas == 0:
                                if DadosEntrada.iloc[dia][i] != 0:
                                    valorVenda = DadosEntrada.iloc[dia][i]
                                    stopLossDescoberto = valorVenda * (1 + stoplossAux)
                                    op = op + 1;
                                    vendas = vendas + 1;

                                    VendidoHoraVenda = hora[i]

                                    HoraCompra = hora[i]
                                    if (plot):
                                        ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='yellowgreen', marker="^")
                                        print('Short Sold: %.2f' % valorVenda)

                        if Negocio == -2:
                            if vendas == 1:
                                ganhoTotal = ganhoTotal + (valorVenda - DadosEntrada.iloc[dia][i])
                                HoraVenda = hora[i]
                                VendidoHoraCompra = hora[i]
                                if (plot):
                                    ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='orange', marker="v")
                                    print('Short Buy: %.2f' % DadosEntrada.iloc[dia][i])
                                vendas = vendas - 1;
                                #valorVendaCompra = DadosDiaSemana.iloc[dia][i] #MUDEI LEMBRAR
                                valorVendaCompra = DadosEntrada.iloc[dia][i]
                                controleCompra = -1

                        if (DadosEntrada.iloc[dia][i] > stopLossDescoberto):
                            if vendas == 1:
                                vendas = vendas - 1
                                valorAtivo = DadosEntrada.iloc[dia][i]
                                ganhoTotal = ganhoTotal + (valorVenda - int(stopLossDescoberto))
                                VendidoHoraCompra = hora[i]
                                HoraVenda = hora[i]
                                if (plot):
                                    ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='orangered', marker="v")
                                    print('Compra Descoberta: %.2f' % DadosEntrada.iloc[dia][i])
                                valorCompraVenda = DadosEntrada.iloc[dia][i]
                                controleCompra = -1

                    if (i == ultimoNegocio):
                        if (compras > 0):
                            ganhoTotal = ganhoTotal + (DadosEntrada.iloc[dia][i] - valorCompra)
                            compras = compras - 1;
                            HoraVenda = hora[i]
                            CompradoHoraVenda = hora[i]

                            if (plot):
                                ax.plot(hora[i], DadosEntrada.iloc[dia][i], color='m', marker="v")
                                print('Sold: %.2f' % DadosEntrada.iloc[dia][i])

                        if vendas != 0:
                            ganhoTotal = ganhoTotal + (valorVenda - DadosEntrada.iloc[dia][i])
                            HoraVenda = hora[i]
                            if (plot):
                                plt.plot(hora[i], DadosEntrada.iloc[dia][i], color='orange', marker="v")
                                print('Sold: %.2f' % DadosEntrada.iloc[dia][i])
                            compras = 0
                            valorVendaCompra = DadosEntrada.iloc[dia][i]
                            VendidoHoraCompra = hora[i]

                # Controla a Entrada dos DAdos para a correlação a cada iteraçao
                posCorrelacao = posCorrelacao + 1

            if (plot):
                plt.show()

        return ganhoTotal, CompradoHoraCompra, CompradoHoraVenda, VendidoHoraCompra, VendidoHoraVenda

    def verificaCompra(Representacao, correlacaoMomento, listaMedias, posicao, ultimoNegocio, plot, comprado, vendido,
                       parametrosNegociacao):
        # print('----------//---------')
        # Parâmetros para compra
        # medidaGanho = 0.002 #Padrão 0.002
        # medidaCorrelacao = 0.8 #Padrão 0.8
        # medidaCorrelacaoRisco = 0.7 #Padrão 0.7
        # margemRisco = 0

        somaGanho = 0
        somaRepresentacao = 0
        somaCorrelacao = 0
        ganho = 0
        clusters = []
        # print(len(correlacaoMomento))

        for i in range(0, len(correlacaoMomento)):
            ganho = listaMedias[i][ultimoNegocio] - listaMedias[i][posicao]

            # if plot:
            # print('Posição atual: %.2f'  %listaMedias[i][posicao])
            # print('Ultima Posição: %.2f'  %listaMedias[i][ultimoNegocio])

            # print('Granho Atual: %.2f'  %ganho)
            # print('Correlação: %.2f'  %correlacaoMomento[i])
            if correlacaoMomento[i] > 0.5:
                # if plot:
                # print('CorrelacaoMomento: ', correlacaoMomento[i])
                # print('ganho: ', ganho)
                # print('Representacao: ', Representacao[i])
                # print('listaMedias[i][posicao]: ', listaMedias[i][posicao])
                # print('listaMedias[i][ultimoNegocio]', listaMedias[i][ultimoNegocio])
                somaCorrelacao = somaCorrelacao + correlacaoMomento[i] * Representacao[i]
                somaRepresentacao = somaRepresentacao + Representacao[i]

                somaGanho = somaGanho + ganho * (Representacao[i] * 1)
                clusters.append(i)

        if (somaGanho != 0) & (somaCorrelacao != 0):
            parametroGanho = somaGanho / somaRepresentacao
            parametroCorrelacao = somaCorrelacao / somaRepresentacao
            # print("Parametro Ganho: ", parametroGanho)
            # print("Parametro Correlacao: ", parametroCorrelacao)
        else:
            parametroGanho = 0
            parametroCorrelacao = 0

        if plot:
            print('ParametroGanho: ', parametroGanho)
            print('parametroCorrelacao: ', parametroCorrelacao)

            print("Clusters: ", clusters)
        if (parametroGanho > parametrosNegociacao['MGanhoPadrao'][0]) & (
                (parametroCorrelacao > parametrosNegociacao['MGanhoCorrelacao'][0]) | (len(clusters) == 1)):
            if plot:
                print('Compra')
            # print('Ganho: ', parametroGanho)
            return 1
        else:
            if (parametroGanho < parametrosNegociacao['MGanhoPadrao'][0]) & (
                    parametroGanho > parametrosNegociacao['MRiscoPadrao'][0]):
                if plot:
                    print('Não Opera')
                return 0
            else:
                # if (parametroGanho < medidaGanho*-1) & (parametroCorrelacao > medidaCorrelacao):
                if (parametroGanho < parametrosNegociacao['MRiscoPadrao'][0]) & (
                        (parametroCorrelacao > parametrosNegociacao['MRiscoCorrelacao'][0]) | (len(clusters) == 1)) & (
                        comprado > 0):
                    if plot:
                        print('Venda')
                    # print('Perda: ', parametroGanho)
                    return -1

        if (parametroGanho < (-1 * parametrosNegociacao['MGanhoPadrao'][0])) & (
                (parametroCorrelacao > parametrosNegociacao['MGanhoCorrelacao'][0]) | (len(clusters) == 1)):
            if plot:
                print('Venda Descoberta')
            return 2
        else:
            if (parametroGanho > parametrosNegociacao['MRiscoPadrao'][0]) & (
                    (parametroCorrelacao > parametrosNegociacao['MRiscoCorrelacao'][0]) | (len(clusters) == 1)) & (
                    vendido > 0):
                if plot:
                    print('Compra Descoberta')
            return -2

        # print('Inconsistente')
        return 3



if __name__ == "__main__":

    #ini = time.time()

    Dados = estruturaDados.executaLeitura("dadosTeste.csv")

    #print(Dados.head())

    #fim = time.time()
    #print("Tempo Leitura: ", fim - ini)

    # Retorna os dados em um período específico
    DadosPeriodo = estruturaDados.periodoDados(Dados, "1/1/2008", "30/12/2018")

    #print(DadosPeriodo.head())

    #fim = time.time()
    #print("Tempo Pesquisa Periodo: ", fim - ini)

    # Lista de Ativos que serão analisados
    ativosAnalisados = ["BBAS3", "ABEV3", "PETR4", "JBSS3"]
    estruturaDados.boxplot(DadosPeriodo, ativosAnalisados)
    #ativosAnalisados = ["ABEV3"]



    #dado1 = Dados[Dados["codigo"] == "ABEV3"].fechamento_atual.values
    #dado2 = pd.to_datetime(Dados[Dados["codigo"] == "ABEV3"].datahora.values)
    #print(dado2)
    #td = pd.Series(dado1, index=dado2)
    #td.plot()

    #dado1 = Dados[Dados["codigo"] == "PETR4"].fechamento_atual.values
    #dado2 = pd.to_datetime(Dados[Dados["codigo"] == "PETR4"].datahora.values)
    #print(dado2)
    #td = pd.Series(dado1, index=dado2)
    #td.plot()


    # Plotar os dados
    #plt.figure()
    #estruturaDados.plot(DadosPeriodo,ativosAnalisados)

    # Atribui a Estrutura Dados somente os ativos a serem analisados
    #Dados = EstruturaDados.estruturaDadosAtivo(DadosEstruturados,ativosAnalisados)

    #EstruturaDados.save_variables(globals())

    teste = estruturaDados.correlacao(DadosPeriodo, ativosAnalisados)

    print(teste.corr('pearson'))


