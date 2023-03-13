from typing import List
import numpy as np 

Matrix = List[List[int]]

__all__ = ['reducedRowEchelonForm', 'rowEchelonForm', 'inverse', 'identity', 'mprint']

    
def reducedRowEchelonForm(self, matrix: Matrix) -> Matrix:
    # Check if that matrix is legitimate (I know it is hard to read but it needs to be fast enough)
    if not all([len(row) == len(matrix[0]) and all([isinstance(i, int) for i in row]) for row in matrix]):
        raise ValueError("Matrix must be legitimate") 
    lastColumnIndex = 0
    for rowIndex in range(len(matrix)):
        while matrix[rowIndex][lastColumnIndex] == 0:
            lastColumnIndex += 1
        row = matrix[rowIndex]
        divider = matrix[rowIndex][lastColumnIndex]
        matrix[rowIndex] = [n / divider for n in row]
        
        for rowIndexInner in range(len(matrix)):
            if rowIndexInner != rowIndex:
                multiplier = matrix[rowIndexInner][lastColumnIndex]
                matrix[rowIndexInner] = [matrix[rowIndexInner][i] - matrix[rowIndex][i] * multiplier for i in range(len(matrix[rowIndexInner]))]
    
    return matrix

def rowEchelonForm(self, matrix: Matrix) -> Matrix:
    # Check if that matrix is legitimate (I know it is hard to read but it needs to be fast enough)
    if not all([len(row) == len(matrix[0]) and all([isinstance(i, int) for i in row]) for row in matrix]):
        raise ValueError("Matrix must be legitimate") 
    lastColumnIndex = 0
    for rowIndex in range(len(matrix)):
        while matrix[rowIndex][lastColumnIndex] == 0:
            lastColumnIndex += 1
        row = matrix[rowIndex]
        divider = matrix[rowIndex][lastColumnIndex]
        matrix[rowIndex] = [n / divider for n in row]
        
        for rowIndexInner in range(len(matrix)):
            if rowIndexInner > rowIndex:
                multiplier = matrix[rowIndexInner][lastColumnIndex]
                matrix[rowIndexInner] = [matrix[rowIndexInner][i] - matrix[rowIndex][i] * multiplier for i in range(len(matrix[rowIndexInner]))]
    
    return matrix

def inverse(self, matrix: Matrix) -> Matrix:
    # Check if that matrix is square and legitimate (I know it is hard to read but it needs to be fast enough)
    if not all([len(row) == len(matrix) and all([isinstance(i, int) for i in row]) for row in matrix]):
        raise ValueError("Matrix must be square and legitimate")
    
    size = len(matrix)
    identityMatrix = self.identity(size)
    matrices = [[matrix[rowIndex] + [identityMatrix[i][rowIndex]] for rowIndex in range(size)] for i in range(size)]
    
    for i in range(size):
        matrices[i] = self.reducedRowEchelonForm(matrices[i]) 
    
    inverseMatrix = [[matrices[h][i][-1:][0] for h in range(size)] for i in range(size)]
    
    return inverseMatrix

@staticmethod
def identity(size: int) -> Matrix:
    if size <= 0:
        raise ValueError("Size must be bigger than zero")
    
    return [[1 if i == h else 0 for h in range(size)] for i in range(size)]

@staticmethod
def mprint(matrix: Matrix) -> None:
    for line in matrix:
        print(line)