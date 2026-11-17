import numpy as np
from dataclasses import dataclass

@dataclass
class SubespacoVetorial:
    
    a: np.ndarray
    
    @property
    def P(self) -> np.ndarray:
        """
        Calcula matriz de projeção no espaço coluna de A.

        Returns
        -------
        np.ndarray
            Matriz de projeção P.
        """
        
        matriz_de_projecao = self.a @ self.a.transpose() @ np.linalg.inv(self.a @ self.a.transpose())
        
        return matriz_de_projecao
    
    
    def projetar(self, b: np.ndarray) -> np.ndarray:
        """
        Projeta o vetor de saída do sistema linear Ax = b no espaço coluna de A.

        Parameters
        ----------
        b : np.ndarray
            Vetor coluna de saída do sistema linear Ax = b.

        Returns
        -------
        np.ndarray
            Vetor b projetado no espaço coluna de A.
        """
        b_corr = self.P @ b
        
        return b_corr
    
    
    def calcular_residuo(self, b: np.ndarray) -> np.ndarray:
        """
        Calcula o vetor de resíduos da solução por mínimos quadrados.

        Parameters
        ----------
        b : np.ndarray
            Vetor coluna de saída do sistema linear Ax = b.

        Returns
        -------
        np.ndarray
            Vetor residual de erros após a projeção.
        """
        
        b_residual = b - self.projetar(b)
        
        return b_residual
    
    
    def norma_erro(self, b: np.ndarray) -> np.ndarray:
        """
        Calcula a norma (magnitude RMS) do erro, ou seja, o módulo do ruído calculado.

        Parameters
        ----------
        b : np.ndarray
            Vetor coluna de saída do sistema linear Ax = b.

        Returns
        -------
        np.ndarray
            Retorna a norma do vetor de resíduos.
        """
        
        norma_de_b_residual = np.linalg.norm(self.calcular_residuo(b))
        
        return norma_de_b_residual
    
    def verifica_ortogonalidade(self, outro_subespaco) -> bool:
        """
        Verifica se os espaços coluna de dois subespaços referidos são ortogonais.

        Parameters
        ----------
        outro_subespaco : SubespacoVetorial
            Outro subespaço vetorial.

        Returns
        -------
        bool
            True se forem ortogonais
            False se não forem ortogonais
        """
        
        produto_interno = self.a.transpose() @ outro_subespaco.a
        linhas, colunas = produto_interno.shape
        
        if np.all(produto_interno == 0):
            return True
        else:
            return False
        
    def validar_propriedades(self):
        """
        Verifica validade da matriz de projeção.

        Returns
        -------
        bool
            True se a matriz de projeção for idempotente e simétrica.
            False se não tiver alguma destas características.
        """
        
        if self.P() == self.P().transpose() and self.P() == np.linalg.matrix_power(self.P(), 2):
            return True
        else:
            return False
        
@dataclass
class CalibrarSensor(SubespacoVetorial):
    
    tensoes_referencia: np.ndarray
    subespaco_modelo: SubespacoVetorial
    
    def ajustar_parametros(self, leituras_reais: np.ndarray) -> np.ndarray:
        """
        Ajusta os parâmetros do ADC, projetando-o no espaço coluna da matriz de coeficientes.

        Parameters
        ----------
        leituras_reais : np.ndarray
            Leituras reais do ADC.

        Returns
        -------
        np.ndarray
            Vetor contendo as leituras corrigidas do ADC.
        """
        
        leitura_supositoria = np.linalg.inv((self.subespaco_modelo)) @ self.subespaco_modelo.transpose() @ leituras_reais
        
        return leitura_supositoria
    
    @property
    def diagnostico_sinal(self, leituras_reais: np.ndarray) -> dict:
        """
        Faz o diagnóstico do sinal obtido.

        Parameters
        ----------
        leituras_reais : np.ndarray
            Leituras reais do ADC.

        Returns
        -------
        dict
            Dicionário contendo informações do diagnóstico.
        """
        
        sinal_filtrado = self.subespaco_modelo.projetar(leituras_reais)
        ruido_isolado = self.subespaco_modelo.calcular_residuo(leituras_reais)
        energia_sinal = self.subespaco_modelo.norma_erro(sinal_filtrado) ** 2
        
        if np.isclose(np.linalg.matrix_power(leituras_reais, 2), energia_sinal, atol = 1.192092896e-07, rtol = 0):
            pitagoras_valido = True
        else:
            pitagoras_valido = False
            
        return {"Sinal Filtrado": sinal_filtrado,
                "Ruido Isolado": ruido_isolado,
                "Energia do sinal": energia_sinal,
                "Pitágoras válido": pitagoras_valido}