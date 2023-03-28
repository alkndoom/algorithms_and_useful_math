import json
import re
import numpy as np
import random

"""
YAPILACAK ŞEYLER:
REDUCE() PARANTEZLİ FONKSİYONLARDA KAFAYI SIYIRIYOR, DÜZELTİLECEK ***
BAYESIAN OPTIMIZASYONU GELECEK
HÜCRELER SADECE SKALARLARLA DEĞİL NUMPY ARRAYLERİYLE DE HESAP YAPABİLECEK
HATA DÜZELTME/KONTROL SİSTEMİ GETİRELECEK
solve value düzelecek
"""


class ByeByeBeautiful:

    @staticmethod
    def load_network(file_name, n_name):
        with open(f'{file_name}', 'r+') as json_file:
            json_dict = json.load(json_file)
            if n_name in json_dict.keys():
                n = ByeByeBeautiful.NeuralNetwork(name=n_name, **json_dict[n_name])
                n.cell_library = json_dict[n_name]['cell_library']
                n.symbols = json_dict[n_name]['symbols']
                n.diff_parameters = json_dict[n_name]['diff_parameters']
                return n
            else:
                raise KeyError('not existing neural network')

    class NeuralNetwork:

        cell_library = {}

        constants = {'e': 2.718281828459045, 'pi': 3.141592653589793}
        symbols = {}
        diff_parameters = {}

        def __init__(self, **kwargs):
            self.name = kwargs['name']
            self.cost = kwargs['cost'] if 'cost' in kwargs else 'mean_squared_error'
            self.optimization = kwargs['optimization'] if 'optimization' in kwargs else 'gradient_descent'
            self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.001
            self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.95
            self.beta = kwargs['beta'] if 'beta' in kwargs else 0.95
            self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.00000000005

        def add_cell(self, cell_list: list = None, **kwargs):
            if cell_list:
                for cell in cell_list:
                    if cell['type'] != 'input':
                        cell_attrs = {'type': cell['type'], 'function': re.sub(r'\s+', '', cell['function']),
                                      'activation': cell['activation'], 'inputs': [], 'outputs': [],
                                      'parameters': [], 'value': 0, 'non_activated_value': 0}
                        m = ' ' + cell_attrs['function'] + ' '
                        while re.search(r'([+\-*/^\s()])[wb]([+\-*/^\s()])', m):
                            if re.search(r'([+\-*/^\s()])[w]([+\-*/^\s()])', m):
                                n = 1
                                while f'w{n}' in self.symbols.keys():
                                    n += 1
                                m = re.sub(r'([+\-*/^\s()])[w]([+\-*/^\s()])', r'\1' + f'w{n}' + r'\2', m, count=1)
                                self.symbols[f'w{n}'] = random.uniform(-2, 2)
                                cell_attrs['parameters'].append(f'w{n}')
                            if re.search(r'([+\-*/^\s()])[b]([+\-*/^\s()])', m):
                                n = 1
                                while f'b{n}' in self.symbols.keys():
                                    n += 1
                                m = re.sub(r'([+\-*/^\s()])[b]([+\-*/^\s()])', r'\1' + f'b{n}' + r'\2', m, count=1)
                                self.symbols[f'b{n}'] = random.uniform(-2, 2)
                                cell_attrs['parameters'].append(f'b{n}')
                        for p in re.findall(r'[+\-*/^\s()]([w]\d+)[+\-*/^\s()]', m):
                            if p not in self.symbols.keys():
                                self.symbols[p] = random.uniform(-2, 2)
                                cell_attrs['parameters'].append(p)
                        for p in re.findall(r'[+\-*/^\s()]([b]\d+)[+\-*/^\s()]', m):
                            if p not in self.symbols.keys():
                                self.symbols[p] = random.uniform(-2, 2)
                                cell_attrs['parameters'].append(p)
                        cell_attrs['function'] = m[1:-1]
                        opl = re.sub(r'[+\-*/^\s()]', ' ', cell_attrs['function']).split()
                        for term in opl:
                            if not (self.isfloat(term) or term[0] in 'wb' or term in self.constants.keys()):
                                cell_attrs['inputs'].append(term)
                        self.symbols[cell['name']] = cell_attrs['function']
                        self.cell_library[cell['name']] = cell_attrs
                    else:
                        cell_attrs = {'type': cell['type'], 'activation': cell['activation'],
                                      'outputs': [], 'value': 0, 'non_activated_value': 0, 'queue': 1}
                        self.symbols[cell['name']] = cell_attrs['value']
                        self.cell_library[cell['name']] = cell_attrs
            else:
                if kwargs['type'] != 'input':
                    cell_attrs = {'type': kwargs['type'], 'function': re.sub(r'\s+', '', kwargs['function']),
                                  'activation': kwargs['activation'], 'inputs': [], 'outputs': [],
                                  'parameters': [], 'value': 0, 'non_activated_value': 0}
                    m = ' ' + cell_attrs['function'] + ' '
                    while re.search(r'([+\-*/^\s()])[wb]([+\-*/^\s()])', m):
                        if re.search(r'([+\-*/^\s()])[w]([+\-*/^\s()])', m):
                            n = 1
                            while f'w{n}' in self.symbols.keys():
                                n += 1
                            m = re.sub(r'([+\-*/^\s()])[w]([+\-*/^\s()])', r'\1' + f'w{n}' + r'\2', m, count=1)
                            self.symbols[f'w{n}'] = random.uniform(-2, 2)
                            cell_attrs['parameters'].append(f'w{n}')
                        if re.search(r'([+\-*/^\s()])[b]([+\-*/^\s()])', m):
                            n = 1
                            while f'b{n}' in self.symbols.keys():
                                n += 1
                            m = re.sub(r'([+\-*/^\s()])[b]([+\-*/^\s()])', r'\1' + f'b{n}' + r'\2', m, count=1)
                            self.symbols[f'b{n}'] = random.uniform(-2, 2)
                            cell_attrs['parameters'].append(f'b{n}')
                    for p in re.findall(r'[+\-*/^\s()]([w]\d+)[+\-*/^\s()]', m):
                        if p not in self.symbols.keys():
                            self.symbols[p] = random.uniform(-2, 2)
                            cell_attrs['parameters'].append(p)
                    for p in re.findall(r'[+\-*/^\s()]([b]\d+)[+\-*/^\s()]', m):
                        if p not in self.symbols.keys():
                            self.symbols[p] = random.uniform(-2, 2)
                            cell_attrs['parameters'].append(p)
                    cell_attrs['function'] = m[1:-1]
                    opl = re.sub(r'[+\-*/^\s()]', ' ', cell_attrs['function']).split()
                    for term in opl:
                        if not (self.isfloat(term) or term[0] in 'wb' or term in self.constants.keys()):
                            cell_attrs['inputs'].append(term)
                    self.symbols[kwargs['name']] = cell_attrs['function']
                    self.cell_library[kwargs['name']] = cell_attrs
                else:
                    cell_attrs = {'type': kwargs['type'], 'activation': kwargs['activation'],
                                  'outputs': [], 'value': 0, 'non_activated_value': 0, 'queue': 1}
                    self.symbols[kwargs['name']] = cell_attrs['value']
                    self.cell_library[kwargs['name']] = cell_attrs
            return self

        # DAMN YOU ARE UGLY | https://www.youtube.com/watch?v=BLYdKDCWe6I
        def start(self):
            self.diff_parameters = {}
            for cell_name, cell_attrs in self.cell_library.items():
                if 'queue' in cell_attrs and cell_attrs['type'] != 'input':
                    del self.cell_library[cell_name]['queue']

            for cell_name, cell_attrs in self.cell_library.items():
                if cell_attrs['type'] != 'input':
                    opl = re.sub(r'[+\-*/^\s()]', ' ', cell_attrs['function']).split()
                    for term in opl:
                        if not (self.isfloat(term) or term[0] in 'wb' or term in self.constants.keys()):
                            term = term[1:] if term[0] == '#' else term
                            if cell_name not in self.cell_library[term]['outputs']:
                                self.cell_library[term]['outputs'].append(cell_name)

            sorted_cell_library = {}
            there_are_items_with_no_queue = True
            while there_are_items_with_no_queue:
                there_are_items_with_no_queue = False
                for cell_name, cell_attrs in self.cell_library.items():
                    if 'queue' not in cell_attrs.keys():
                        there_are_items_with_no_queue = True
                        biggest_queue = 0
                        dependencies = [a for a in cell_attrs['inputs'] if a[0] != '#']
                        every_item_has_queue = True
                        for input_cell_name in dependencies:
                            if 'queue' not in self.cell_library[input_cell_name].keys():
                                every_item_has_queue = False
                        if every_item_has_queue:
                            for input_cell_name in dependencies:
                                if biggest_queue < self.cell_library[input_cell_name]['queue']:
                                    biggest_queue = self.cell_library[input_cell_name]['queue']
                                    cell_attrs['queue'] = biggest_queue + 1
            ct = 1
            flag3 = True
            while flag3:
                flag3 = False
                for cell_name, cell_attrs in self.cell_library.items():
                    if cell_attrs['queue'] == ct:
                        flag3 = True
                        sorted_cell_library.update({cell_name: cell_attrs})
                ct += 1
            self.cell_library = sorted_cell_library

            for cell_name, cell_attrs in self.cell_library.items():
                if cell_attrs['type'] != 'input':
                    for i in cell_attrs['inputs']:
                        if i[0] != '#' and self.cell_library[i]['type'] != 'input':
                            for j in self.cell_library[i]['parameters']:
                                if j not in cell_attrs['parameters']:
                                    self.cell_library[cell_name]['parameters'].append(j)

            def diff_activation(forward_cell, parameter):
                activations_to_use = []
                queue = [[forward_cell]]
                while queue:
                    current_path = queue.pop()
                    current_cell = current_path[-1:][0]
                    if self.cell_library[current_cell]['type'] == 'input':
                        pass
                    elif parameter in self.cell_library[current_cell]['function']:
                        activations_to_use.append(current_path)
                    else:
                        for input_ in [a for a in self.cell_library[current_cell]['inputs'] if a[0] != '#']:
                            lp = current_path + [input_]
                            queue.append(lp)
                c_a_t_u = []
                for cells in activations_to_use:
                    c_a_t_u += cells
                activations_to_use = [a for a in c_a_t_u if self.cell_library[a]['activation'] != 'identity']
                return activations_to_use

            output_cells = [cell_name for cell_name, cell_attrs in self.cell_library.items() if cell_attrs['type'] == 'output']
            for param in [symb for symb in self.symbols.keys() if symb[0] in 'wb']:
                for o_cell in output_cells:
                    if param in self.cell_library[o_cell]['parameters']:
                        if f'{o_cell}|{param}' not in self.diff_parameters.keys():
                            activation_list = diff_activation(o_cell, param)
                            diff = self.differentiate(o_cell, param)
                            for _ in range(10):
                                diff = self.reduce(diff)
                            self.diff_parameters[f'{o_cell}|{param}'] = [diff, activation_list]
            return self

        def script(self, file_name):
            code = """import numpy as np
import matplotlib.pyplot as plt
import re
import json
import random


class Activations:

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def binary_step(x):
        return 0 if x <= 0 else 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return 0 if x <= 0 else x

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def selu(x, alpha=1.0507, beta=1.67326):
        return alpha * beta * (np.exp(x) - 1) if x < 0 else alpha * x

    @staticmethod
    def prelu(x, alpha=0.01):
        return alpha * x if x < 0 else x

    @staticmethod
    def silu(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    def gaussian(x):
        return np.exp(-x ** 2)


class DifferentialActivations:

    @staticmethod
    def identity(x):
        return 1

    @staticmethod
    def binary_step(x):
        if x != 0:
            return 0
        else:
            raise ValueError("function is not differentiable at zero")

    @staticmethod
    def sigmoid(x):
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

    @staticmethod
    def tanh(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        else:
            raise ValueError("function is not differentiable at zero")

    @staticmethod
    def softplus(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def selu(x, alpha=1.0507, beta=1.67326):
        return alpha * beta * np.exp(x) if x < 0 else alpha

    @staticmethod
    def prelu(x, alpha=0.01):
        return alpha if x < 0 else 1

    @staticmethod
    def silu(x):
        return (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x))**2

    @staticmethod
    def gaussian(x):
        return -2 * x * np.exp(-x**2)


"""

            class_name = ''.join([word.capitalize() for word in re.split(r'[ _]', file_name)])
            code += f'class {class_name}:\n\n'

            # HYPERPARAMETERS
            code += f"    name = '{self.name}'\n"
            code += f"    cost = '{self.cost}'\n"
            code += f"    optimization = '{self.optimization}'\n"
            code += f'    learning_rate = {self.learning_rate}\n'
            code += f'    alpha = {self.alpha}\n'
            code += f'    beta = {self.beta}\n'
            code += f'    epsilon = {self.epsilon}\n'

            code += """
    TEMP_DATA = [[[x], [3 * x]] for x in range(1, 100)]

    error_history = []

    activations = {
        'identity': Activations.identity,
        'binary_step': Activations.binary_step,
        'sigmoid': Activations.sigmoid,
        'tanh': Activations.tanh,
        'relu': Activations.relu,
        'softplus': Activations.softplus,
        'selu': Activations.selu,
        'prelu': Activations.prelu,
        'silu': Activations.silu,
        'gaussian': Activations.gaussian,
    }

    differential_activations = {
        'identity': DifferentialActivations.identity,
        'binary_step': DifferentialActivations.binary_step,
        'sigmoid': DifferentialActivations.sigmoid,
        'tanh': DifferentialActivations.tanh,
        'relu': DifferentialActivations.relu,
        'softplus': DifferentialActivations.softplus,
        'selu': DifferentialActivations.selu,
        'prelu': DifferentialActivations.prelu,
        'silu': DifferentialActivations.silu,
        'gaussian': DifferentialActivations.gaussian,
    }

    costs = {
        'mean_error': lambda theta, target: theta - target,
        'mean_squared_error': lambda theta, target: (theta - target) ** 2,
        'mean_absolute_error': lambda theta, target: np.abs(theta - target),
        'cross_entropy': lambda theta, target: sum([np.log(theta[i]) * target[i] for i in range(len(theta))]),
    }

    differential_costs = {
        'mean_error': lambda theta, target: 1,
        'mean_squared_error': lambda theta, target: 2 * (theta - target),
        'mean_absolute_error': lambda theta, target: 1 if theta >= target else 0,
        'cross_entropy': lambda theta, target: sum([np.log(theta[i]) * target[i] for i in range(len(theta))]),
    }

"""

            # SYMBOLS
            symb = '    symbols = {\n'
            for symbol_name, symbol_function in self.symbols.items():
                if symbol_name in self.cell_library.keys():
                    symb += f"        '{symbol_name}': '{symbol_function}',\n"
                else:
                    symb += f"        '{symbol_name}': {symbol_function},\n"
            symb += '    }\n\n'
            code += symb

            # DIFFERENTIAL PARAMETERS
            diff_p = '    diff_parameters = {\n'
            for parameter_name, parameter_function in self.diff_parameters.items():
                diff_function = ' ' + parameter_function[0] + ' '
                for symbol in self.symbols.keys():
                    if re.search(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", diff_function):
                        if symbol in diff_function:
                            if symbol in self.cell_library.keys():
                                diff_function = re.sub(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", r"\1" + f"self.cell_library['{symbol}']['value']" + r"\2", diff_function)
                            else:
                                diff_function = re.sub(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", r"\1" + f"self.symbols['{symbol}']" + r"\2", diff_function)
                diff_function = diff_function[1:-1]
                diff_function = re.sub('\^', '**', diff_function)
                diff_function = re.sub('#', '', diff_function)
                for activation_value in parameter_function[1]:
                    act = self.cell_library[activation_value]['activation']
                    diff_function += f"*self.differential_activations['{act}'](self.cell_library['{activation_value}']['non_activated_value'])"
                diff_p += f"        '{parameter_name}': lambda self: {diff_function},\n"
            diff_p += '    }\n\n'
            code += diff_p

            # CELL LIBRARY
            c_l = '    cell_library = {\n'
            for cell_name, cell_attrs in self.cell_library.items():
                c_l += f"        '{cell_name}': " + '{\n'
                for attr_name, attr_ in cell_attrs.items():
                    if attr_name != 'function':
                        if type(attr_) == str:
                            c_l += f"            '{attr_name}': '{attr_}',\n"
                        else:
                            c_l += f"            '{attr_name}': {attr_},\n"
                    else:
                        attr_ = ' ' + attr_ + ' '
                        for symbol in self.symbols.keys():
                            if re.search(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", attr_):
                                if symbol in self.cell_library.keys():
                                    attr_ = re.sub(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", r"\1" + f"self.cell_library['{symbol}']['value']" + r"\2", attr_)
                                else:
                                    attr_ = re.sub(rf"([+\-*/^\s()]){symbol}([+\-*/^\s()])", r"\1" + f"self.symbols['{symbol}']" + r"\2", attr_)
                        attr_ = attr_[1:-1]
                        attr_ = re.sub('\^', '**', attr_)
                        attr_ = re.sub('#', '', attr_)
                        c_l += f"            '{attr_name}': lambda self: {attr_},\n"
                c_l += '        },\n'
            c_l += '    }\n'
            code += c_l

            code += """
    def __init__(self):
        self.output_cells = [cell_name for cell_name, cell_attrs in self.cell_library.items() if cell_attrs['type'] == 'output']
        self.input_cells = [cell_name for cell_name, cell_attrs in self.cell_library.items() if cell_attrs['type'] == 'input']
        if self.optimization == 'gradient_descent':
            self.optimize = self.gradient_descent
        elif self.optimization == 'momentum':
            self.optimize = self.momentum
        elif self.optimization == 'adagrad':
            self.optimize = self.adagrad
        elif self.optimization == 'rmsprop':
            self.optimize = self.rmsprop
        elif self.optimization == 'adam':
            self.optimize = self.adam
        else:
            self.optimize = self.gradient_descent

    def feed_forward(self, data_set):
        iter_input = data_set[0][0]
        target = data_set[1]

        for cell_name, cell_attrs in self.cell_library.items():
            if cell_attrs['type'] == 'input':
                self.cell_library[cell_name]['non_activated_value'] = iter_input
                self.cell_library[cell_name]['value'] = iter_input
        for cell_name, cell_attrs in self.cell_library.items():
            if cell_attrs['type'] != 'input':
                self.cell_library[cell_name]['non_activated_value'] = cell_attrs['function'](self)
                self.cell_library[cell_name]['value'] = self.activations[cell_attrs['activation']](cell_attrs['function'](self))

        output_values = [self.cell_library[cell_name]['value'] for cell_name in self.output_cells]
        if self.cost != 'cross_entropy':
            self.error_history.append(self.costs[self.cost](output_values[0], target[0]))
        else:
            self.error_history.append(self.costs[self.cost](output_values, target))

    def back_propagation(self, data_set):
        target = data_set[1]

        output_values = [self.cell_library[cell_name]['value'] for cell_name in self.output_cells]
        if self.cost != 'cross_entropy':
            diff_cost = self.differential_costs[self.cost](output_values[0], target[0])
        else:
            diff_cost = self.differential_costs[self.cost](output_values, target)

        print('COST: %0.25f' % (diff_cost,))
        if abs(diff_cost) < 0.000001 or abs(diff_cost) > 1000:
            print('********************STOPPED********************')
            return 0

        for diff_param in self.diff_parameters.keys():
            diff_param = diff_param.split('|')
            o_cell = diff_param[0]
            param = diff_param[1]
            diff = self.diff_parameters[f'{o_cell}|{param}'](self) * diff_cost
            self.symbols[param] = self.optimize(self.symbols[param], diff)
        return self

    def train(self, data_set, epochs=1):
        for _ in range(epochs):
            for data in data_set:
                self.feed_forward(data)
                if self.back_propagation(data) == 0:
                    break

    def predict(self, data_set):
        pass
        
    def save_network(self, file_name, indent=2, **kwargs):
        file_name = re.sub('\s', '_', file_name.lower())
        try:
            open(f'{file_name}.json')
        except IOError:
            with open(f'{file_name}.json', 'w') as json_file:
                json_file.write('{}')

        with open(f'{file_name}.json', 'r+') as json_file:
            json_dict = json.load(json_file)
            json_dict[self.name]['cost'] = self.cost
            json_dict[self.name]['optimization'] = self.optimization
            json_dict[self.name]['learning_rate'] = self.learning_rate
            json_dict[self.name]['alpha'] = self.alpha
            json_dict[self.name]['beta'] = self.beta
            json_dict[self.name]['epsilon'] = self.epsilon
            json_dict[self.name]['symbols'] = self.symbols
            for cell_name, cell_attrs in self.cell_library.items():
                json_dict[self.name]['cell_library'][cell_name]['value'] = cell_attrs['value']
                json_dict[self.name]['cell_library'][cell_name]['non_activated_value'] = cell_attrs['non_activated_value']
            json_file.seek(0)
            json.dump(json_dict, json_file, indent=indent, **kwargs)
        return self
    
    def load_network(self, file_name):
        with open(f'{file_name}', 'r+') as json_file:
            json_dict = json.load(json_file)
            if self.name in json_dict.keys():
                self.symbols = json_dict[self.name]['symbols']
                self.cost = json_dict[self.name]['cost']
                self.optimization = json_dict[self.name]['optimization']
                self.learning_rate = json_dict[self.name]['learning_rate']
                self.alpha = json_dict[self.name]['alpha']
                self.beta = json_dict[self.name]['beta']
                self.epsilon = json_dict[self.name]['epsilon']
                return self
            else:
                raise KeyError('not existing neural network')

    t = 0  # Iteration
    v = 0  # V Initial
    z = 0  # Z Initial
    s = 0  # S Initial

    def gradient_descent(self, theta, diff):
        self.t += 1
        theta -= self.learning_rate * diff
        return theta

    def momentum(self, theta, diff):
        self.t += 1
        self.v = self.alpha * self.v + (1 - self.alpha) * diff
        theta -= self.learning_rate * self.v
        return theta

    def adagrad(self, theta, diff):
        self.t += 1
        self.z += diff ** 2
        theta -= self.learning_rate / (np.sqrt(self.z) + self.epsilon) * diff
        return theta

    def rmsprop(self, theta, diff):
        self.t += 1
        self.s = self.alpha * self.s + (1 - self.alpha) * diff ** 2
        theta -= self.learning_rate / (np.sqrt(self.s) + self.epsilon) * diff
        return theta

    def adam(self, theta, diff):
        self.t += 1
        self.v = self.alpha * self.v + (1 - self.alpha) * diff
        self.s = self.beta * self.s + (1 - self.beta) * diff ** 2
        v_prime = self.v / (1 - self.alpha ** self.t)
        s_prime = self.s / (1 - self.beta ** self.t)
        theta -= self.learning_rate / (np.sqrt(s_prime) + self.epsilon) * v_prime
        return theta
    

n1 = TryNumpy()
print(f'Symbols: {n1.symbols}')
for c_name, c_attrs in n1.cell_library.items():
    print(f'{c_name}: {c_attrs}')

if n1.error_history:
    plt.plot(range(1, len(n1.error_history) + 1), n1.error_history)
    plt.show()

"""

            with open(f'{file_name}.py', 'w') as f:
                for line in code.split('\n'):
                    f.write(f'{line}\n')
            return self

        def save_network(self, file_name, indent=2, **kwargs):
            file_name = re.sub('\s', '_', file_name.lower())
            try:
                open(f'{file_name}.json')
            except IOError:
                with open(f'{file_name}.json', 'w') as json_file:
                    json_file.write('{}')

            with open(f'{file_name}.json', 'r+') as json_file:
                json_dict = json.load(json_file)
                json_dict[self.name] = {'cost': self.cost, 'optimization': self.optimization, 'learning_rate': self.learning_rate, 'alpha': self.alpha,
                                        'beta': self.beta, 'epsilon': self.epsilon, 'cell_library': self.cell_library, 'symbols': self.symbols,
                                        'diff_parameters': self.diff_parameters}
                json_file.seek(0)
                json.dump(json_dict, json_file, indent=indent, **kwargs)
            return self

        # ********************TOOLKIT********************
        @staticmethod
        def isfloat(x):
            try:
                float(x)
            except ValueError:
                return False
            else:
                return True

        @staticmethod
        def initialize_equation(equation: str):
            last_read = ''
            capture = True
            summation = []
            for char in equation:
                if char == 'e':
                    last_read += char
                    capture = False
                elif char == '+' and capture:
                    summation.append(last_read)
                    last_read = ''
                else:
                    last_read += char
                    capture = True
            summation.append(last_read)
            for sum_ in range(len(summation)):
                last_read = ''
                capture = True
                substraction = []
                for char in summation[sum_]:
                    if char in 'e^*/':
                        last_read += char
                        capture = False
                    elif char == '-' and capture:
                        substraction.append(last_read)
                        last_read = ''
                    else:
                        last_read += char
                        capture = True
                substraction.append(last_read)
                summation[sum_] = substraction
            for sum_ in range(len(summation)):
                for sub in range(len(summation[sum_]) - 1):
                    summation[sum_][sub + 1] = '-1*' + summation[sum_][sub + 1]
                if summation[sum_][0] == '':
                    del summation[sum_][0]
            summation = [sub for sum_ in summation for sub in sum_]
            for sum_ in range(len(summation)):
                summation[sum_] = summation[sum_].split('*')
            for sum_ in range(len(summation)):
                for mul in range(len(summation[sum_])):
                    summation[sum_][mul] = summation[sum_][mul].split('/')
            for sum_ in range(len(summation)):
                for mul in range(len(summation[sum_])):
                    for div in range(len(summation[sum_][mul]) - 1):
                        summation[sum_][mul][div + 1] = summation[sum_][mul][div + 1] + '^-1'
                summation[sum_] = [div for mul in summation[sum_] for div in mul]
            for sum_ in range(len(summation)):
                for mul in range(len(summation[sum_])):
                    summation[sum_][mul] = summation[sum_][mul].split('^')
            return summation

        def differentiate(self, equation: str, x: str):
            if self.isfloat(equation) or equation in self.constants.keys():  # If it is a constant
                return '0'
            elif equation == x:  # If it is an identity
                return '1'
            elif equation in self.symbols.keys():
                return self.differentiate(self.symbols[equation], x)
            elif equation[0] == '#':
                return '0'
            else:
                equation, p_dict = self.find_all_outer_parentheses(equation)
                equation_list = self.initialize_equation(equation)

                def diff_pow(mul: list, x: str):
                    # DURUM 1 f
                    for char in range(len(mul)):
                        if 'p_' in mul[char]:
                            mul[char] = p_dict[mul[char]]
                    if len(mul) == 1:
                        t = f'{self.differentiate(mul[0], x)}'
                    else:
                        bot = self.isfloat(mul[0]) or mul[0] in self.constants.keys()
                        top = self.isfloat(mul[1]) or mul[1] in self.constants.keys()
                        # DURUM 2 c^c
                        if bot and top:
                            t = f'0'
                        # DURUM 3 u^c
                        elif not bot and top:
                            t = f'{self.differentiate(mul[0], x)}*{mul[1]}*({mul[0]})^{float(mul[1]) - 1}'
                        # DURUM 4 c^u
                        elif bot and not top:
                            t = f'{self.differentiate(mul[1], x)}*({mul[0]})^({mul[1]})*ln({mul[0]})'
                        # DURUM 5 u^u
                        else:
                            t = f'({mul[0]})^({mul[1]})*({self.differentiate(mul[0], x)}*({mul[1]})/({mul[0]})+{self.differentiate(mul[1], x)}*ln({mul[0]}))'
                    return f'{t}'

                def diff_multiply(sum_: list, x: str):
                    if len(sum_) == 1:
                        first = sum_[0]
                        for char in range(len(first)):
                            if 'p_' in first[char]:
                                first[char] = p_dict[first[char]]
                        t1 = diff_pow(first, x)
                        return f'({t1})'
                    elif len(sum_) == 2:
                        first, second = sum_[0], sum_[1]
                        for char in range(len(first)):
                            if 'p_' in first[char]:
                                first[char] = p_dict[first[char]]
                        for char in range(len(second)):
                            if 'p_' in second[char]:
                                second[char] = p_dict[second[char]]
                        t1 = diff_pow(first, x)
                        t4 = f'{first[0]}' if len(first) == 1 else f'{first[0]}^({first[1]})'
                        t3 = diff_pow(second, x)
                        t2 = f'{second[0]}' if len(second) == 1 else f'{second[0]}^({second[1]})'
                        return f'(({t1})*({t2})+({t3})*({t4}))'
                    else:
                        first, second = sum_[:-1], sum_[-1:][0]
                        for char in range(len(second)):
                            if 'p_' in second[char]:
                                second[char] = p_dict[second[char]]
                        product = [f'{mul[0]}' if len(mul) == 1 else f'{mul[0]}^{mul[1]}' for mul in first]
                        t1 = diff_multiply(first, x)
                        t4 = '*'.join(product)
                        t3 = diff_pow(second, x)
                        t2 = f'{second[0]}' if len(second) == 1 else f'{second[0]}^({second[1]})'
                        return f'(({t1})*({t2})+({t3})*({t4}))'

                summation = []
                for sum_ in range(len(equation_list)):
                    summation.append(diff_multiply(equation_list[sum_], x))
                all_sum = '+'.join(summation)
                return re.sub('--', '+', re.sub(r'\+-', '-', f'({all_sum})'))

        @staticmethod
        def find_all_outer_parentheses(equation: str):
            equation = re.sub(r'\s+', '', equation)
            p_dict = {}
            if any(char in equation for char in '()'):
                p_id = 0
                clear_equation = ''
                last_read = ''
                capture = True
                parenthesis_count = 0
                for i in range(len(equation)):
                    if equation[i] == '(':
                        parenthesis_count += 1
                        if capture:
                            capture = False
                            clear_equation += last_read
                            last_read = ''
                        else:
                            last_read += '('
                    elif equation[i] == ')':
                        parenthesis_count -= 1
                        if parenthesis_count == 0:
                            capture = True
                            p_name = f'p_{p_id}'
                            p_id += 1
                            p_dict[p_name] = last_read
                            clear_equation += p_name
                            last_read = ''
                        else:
                            last_read += ')'
                    else:
                        last_read += equation[i]
                clear_equation += last_read
            else:
                clear_equation = equation
            return clear_equation, p_dict

        def value(self, x):
            if self.isfloat(x):
                a = float(x)
            elif x[0] == '#':
                a = self.cell_library[x[1:]]['output']
            elif x in self.constants.keys():
                a = self.constants[x]
            elif 'log' in x:
                a = np.log(self.value(x[3:]), 10)
            elif 'ln' in x:
                a = np.log(self.value(x[2:]))
            elif 'sin' in x:
                a = np.sin(self.value(x[3:]))
            elif 'cos' in x:
                a = np.cos(self.value(x[3:]))
            elif 'tan' in x:
                a = np.tan(self.value(x[3:]))
            elif 'cot' in x:
                a = np.cot(self.value(x[3:]))
            elif 'sec' in x:
                a = 1 / np.cos(self.value(x[3:]))
            elif 'csc' in x:
                a = 1 / np.sin(self.value(x[3:]))
            elif 'sinh' in x:
                a = np.sinh(self.value(x[4:]))
            elif 'cosh' in x:
                a = np.cosh(self.value(x[4:]))
            elif 'tanh' in x:
                a = np.tanh(self.value(x[4:]))
            elif 'coth' in x:
                a = np.coth(self.value(x[4:]))
            elif 'sech' in x:
                a = 1 / np.cosh(self.value(x[4:]))
            elif 'csch' in x:
                a = 1 / np.sinh(self.value(x[4:]))
            elif 'arcsin' in x:
                a = np.arcsin(self.value(x[6:]))
            elif 'arccos' in x:
                a = np.arccos(self.value(x[6:]))
            elif 'arctan' in x:
                a = np.arctan(self.value(x[6:]))
            elif 'arccot' in x:
                a = np.arccot(self.value(x[6:]))
            elif 'arcsec' in x:
                a = np.arccos(1 / self.value(x[6:]))
            elif 'arccsc' in x:
                a = np.arcsin(1 / self.value(x[6:]))
            elif 'arsinh' in x:
                a = np.arcsinh(self.value(x[6:]))
            elif 'arcosh' in x:
                a = np.arccosh(self.value(x[6:]))
            elif 'artanh' in x:
                a = np.arctanh(self.value(x[6:]))
            elif 'arcoth' in x:
                a = np.arccoth(self.value(x[6:]))
            elif 'arsech' in x:
                a = np.arccosh(1 / self.value(x[6:]))
            elif 'arcsch' in x:
                a = np.arcsinh(1 / self.value(x[6:]))
            elif x in self.cell_library.keys():
                a = self.cell_library[x]['value']
            else:
                a = self.solve(f'{self.symbols[x]}')
            return a

        def solve(self, equation: str):
            while any(char in equation for char in '()'):
                start_index, end_index = 0, 0
                last_read = ''
                for j, i in enumerate(equation):
                    if i == '(':
                        start_index = j
                        last_read = ''
                    elif i == ')':
                        end_index = j
                        break
                    else:
                        last_read += i
                equation = re.sub(r'\+-', '-', f'{equation[:start_index]}{self.solve(last_read)}{equation[end_index + 1:]}')
                equation = re.sub(r'--', '+', equation)
            equation_list = self.initialize_equation(equation)
            summation = 0
            for sum_ in range(len(equation_list)):
                product = 1
                for mul in range(len(equation_list[sum_])):
                    if len(equation_list[sum_][mul]) == 2:
                        product *= self.value(equation_list[sum_][mul][0]) ** self.value(equation_list[sum_][mul][1])
                    else:
                        product *= self.value(equation_list[sum_][mul][0])
                summation += product
            return summation

        def reduce(self, equation: str, unreducable=None):
            if not unreducable:
                unreducable = {}
            equation, p_dict = self.find_all_outer_parentheses(equation)
            for p_name, p_attrs in p_dict.items():
                start = re.findall(r'^\(+', p_attrs)
                end = re.findall(r'\)+$', p_attrs)
                if start and end:
                    if len(start[0]) > 1 and len(end[0]) > 1:
                        m = min(len(start[0]), len(end[0])) - 1
                        p_dict[p_name] = p_attrs[m:-m]
            summation = self.initialize_equation(equation)
            for sum_ in range(len(summation)):
                for mul in range(len(summation[sum_])):
                    pows = summation[sum_][mul]
                    if len(pows) == 1:
                        summation[sum_][mul] = [pows[0]]
                    elif len(pows) == 2:
                        if pows[0] == '1':
                            summation[sum_][mul] = ['1']
                        elif pows[0] == '0' and pows[1] != '0':
                            summation[sum_][mul] = ['0']
                        elif pows[0] == '0' and pows[1] == '0':
                            raise ValueError('undefined value: 0^0')
                        elif pows[1] == '1':
                            summation[sum_][mul] = [pows[0]]
                        elif pows[1] == '0':
                            summation[sum_][mul] = ['1']
                        else:
                            if self.isfloat(pows[0]) and self.isfloat(pows[1]):
                                summation[sum_][mul] = [f'{float(pows[0]) ** float(pows[1])}']
                            else:
                                summation[sum_][mul] = pows
                    else:
                        product = 1
                        non_product = ''
                        for pow_ in pows[1:]:
                            if self.isfloat(pow_):
                                product *= float(pow_)
                            else:
                                non_product += f'*{pow_}'
                        pows = [pows[0]] + [f'{product}' + non_product]
                        if pows[0] == '1':
                            summation[sum_][mul] = ['1']
                        elif pows[0] == '0' and pows[1] != '0':
                            summation[sum_][mul] = ['0']
                        elif pows[0] == '0' and pows[1] == '0':
                            raise ValueError('undefined value: 0^0')
                        elif pows[1] == '1':
                            summation[sum_][mul] = [pows[0]]
                        elif pows[1] == '0':
                            summation[sum_][mul] = ['1']
                        else:
                            if self.isfloat(pows[0]) and self.isfloat(pows[1]):
                                summation[sum_][mul] = [f'{float(pows[0]) ** float(pows[1])}']
                            else:
                                summation[sum_][mul] = pows

            reduced_multiplication = []
            for sum_ in summation:
                if any(mul[0] for mul in sum_ if mul[0] == '0'):
                    reduced_multiplication.append([['0']])
                else:
                    mull = []
                    for mul in sum_:
                        if len(sum_) > 1:
                            if mul[0] != '1':
                                mull.append(mul)
                        else:
                            mull.append(mul)
                    reduced_multiplication.append(mull)
            reduced_summation = []
            for sum_ in reduced_multiplication:
                if sum_ != [['0']]:
                    reduced_summation.append(sum_)
            reduced_equation = []
            for sum_ in range(len(reduced_summation)):
                product = []
                for mul in range(len(reduced_summation[sum_])):
                    for pow_ in range(len(reduced_summation[sum_][mul])):
                        if 'p_' in reduced_summation[sum_][mul][pow_]:
                            p_key = re.search('(p_\d+)', reduced_summation[sum_][mul][pow_]).group()
                            no_parenthesis = self.reduce(p_dict[p_key], unreducable)
                            if len(re.split('[+\-*/^]', no_parenthesis)) > 1:
                                n = 0
                                while f'ur_{n}' in unreducable.keys():
                                    n += 1
                                unreducable[f'ur_{n}'] = no_parenthesis
                                reduced_summation[sum_][mul][pow_] = re.sub(p_key, f'ur_{n}', reduced_summation[sum_][mul][pow_])
                            else:
                                reduced_summation[sum_][mul][pow_] = re.sub(p_key, no_parenthesis, reduced_summation[sum_][mul][pow_])
                    if len(reduced_summation[sum_][mul]) == 1:
                        bot = reduced_summation[sum_][mul][0]
                        if len(bot.split('+')) > 1:
                            bot = f'({bot})'
                        product.append(f'{bot}')
                    elif len(reduced_summation[sum_][mul]) == 2:
                        bot = reduced_summation[sum_][mul][0]
                        top = reduced_summation[sum_][mul][1]
                        if len(bot.split('+')) > 1:
                            bot = f'({bot})'
                        if len(top.split('+')) > 1:
                            top = f'({top})'
                        product.append(f'{bot}^{top}')
                reduced_equation.append('*'.join(product))
            for term in range(len(reduced_equation)):
                while 'ur_' in reduced_equation[term]:
                    reduced_equation[term] = re.sub(r'(ur_\d+)', lambda x: rf'({unreducable[x.group()]})', reduced_equation[term])
            if len(reduced_equation) == 0:
                return '0'
            else:
                ret = '+'.join(reduced_equation)
                start = re.findall(r'^\(+', ret)
                end = re.findall(r'\)+$', ret)
                if start and end:
                    if len(start[0]) > 1 and len(end[0]) > 1:
                        m = min(len(start[0]), len(end[0])) - 1
                        ret = ret[m:-m]
                return ret


"""n1 = ByeByeBeautiful.NeuralNetwork(name='n1', optimization='momentum', alpha=0.8, beta=0.85)

n1.add_cell([{'name': 'x', 'type': 'input', 'function': 0, 'activation': 'identity'},
             {'name': 'h', 'type': 'hidden', 'function': '(w*x)^3', 'activation': 'identity'},
             {'name': 'c', 'type': 'output', 'function': 'w*h+b', 'activation': 'identity'}])

n1.start().save_network('Try Numpy').script('Try Numpy')"""

"""n1 = ByeByeBeautiful.load_network('try_numpy.json', 'n1')

print(n1.cell_library)"""

n1 = ByeByeBeautiful.NeuralNetwork(name='n1', optimization='momentum', alpha=0.8, beta=0.85)

n1.add_cell([{'name': 'i0', 'type': 'input', 'activation': 'identity'},
             {'name': 'i1', 'type': 'input', 'activation': 'identity'},
             {'name': 'i2', 'type': 'input', 'activation': 'identity'},
             {'name': 'i3', 'type': 'input', 'activation': 'identity'},
             {'name': 'i4', 'type': 'input', 'activation': 'identity'},
             {'name': 'i5', 'type': 'input', 'activation': 'identity'},
             {'name': 'i6', 'type': 'input', 'activation': 'identity'},
             {'name': 'i7', 'type': 'input', 'activation': 'identity'},
             {'name': 'i8', 'type': 'input', 'activation': 'identity'},
             {'name': 'i9', 'type': 'input', 'activation': 'identity'},
             {'name': 'i10', 'type': 'input', 'activation': 'identity'},
             {'name': 'i11', 'type': 'input', 'activation': 'identity'},
             {'name': 'i12', 'type': 'input', 'activation': 'identity'},
             {'name': 'i13', 'type': 'input', 'activation': 'identity'},
             {'name': 'i14', 'type': 'input', 'activation': 'identity'},
             {'name': 'i15', 'type': 'input', 'activation': 'identity'},
             {'name': 'i16', 'type': 'input', 'activation': 'identity'},
             {'name': 'i17', 'type': 'input', 'activation': 'identity'},
             {'name': 'i18', 'type': 'input', 'activation': 'identity'},
             {'name': 'i19', 'type': 'input', 'activation': 'identity'},
             {'name': 'i20', 'type': 'input', 'activation': 'identity'},
             {'name': 'i21', 'type': 'input', 'activation': 'identity'},
             {'name': 'i22', 'type': 'input', 'activation': 'identity'},
             {'name': 'i23', 'type': 'input', 'activation': 'identity'},
             {'name': 'i24', 'type': 'input', 'activation': 'identity'},
             {'name': 'i25', 'type': 'input', 'activation': 'identity'},
             {'name': 'i26', 'type': 'input', 'activation': 'identity'},
             {'name': 'i27', 'type': 'input', 'activation': 'identity'},
             {'name': 'i28', 'type': 'input', 'activation': 'identity'},
             {'name': 'i29', 'type': 'input', 'activation': 'identity'},
             {'name': 'i30', 'type': 'input', 'activation': 'identity'},
             {'name': 'i31', 'type': 'input', 'activation': 'identity'},
             {'name': 'i32', 'type': 'input', 'activation': 'identity'},
             {'name': 'i33', 'type': 'input', 'activation': 'identity'},
             {'name': 'i34', 'type': 'input', 'activation': 'identity'},
             {'name': 'i35', 'type': 'input', 'activation': 'identity'},
             {'name': 'i36', 'type': 'input', 'activation': 'identity'},
             {'name': 'i37', 'type': 'input', 'activation': 'identity'},
             {'name': 'i38', 'type': 'input', 'activation': 'identity'},
             {'name': 'i39', 'type': 'input', 'activation': 'identity'},
             {'name': 'i40', 'type': 'input', 'activation': 'identity'},
             {'name': 'i41', 'type': 'input', 'activation': 'identity'},
             {'name': 'i42', 'type': 'input', 'activation': 'identity'},
             {'name': 'i43', 'type': 'input', 'activation': 'identity'},
             {'name': 'i44', 'type': 'input', 'activation': 'identity'},
             {'name': 'i45', 'type': 'input', 'activation': 'identity'},
             {'name': 'i46', 'type': 'input', 'activation': 'identity'},
             {'name': 'i47', 'type': 'input', 'activation': 'identity'},
             {'name': 'i48', 'type': 'input', 'activation': 'identity'},
             {'name': 'i49', 'type': 'input', 'activation': 'identity'},
             {'name': 'i50', 'type': 'input', 'activation': 'identity'},
             {'name': 'i51', 'type': 'input', 'activation': 'identity'},
             {'name': 'i52', 'type': 'input', 'activation': 'identity'},
             {'name': 'i53', 'type': 'input', 'activation': 'identity'},
             {'name': 'i54', 'type': 'input', 'activation': 'identity'},
             {'name': 'i55', 'type': 'input', 'activation': 'identity'},
             {'name': 'i56', 'type': 'input', 'activation': 'identity'},
             {'name': 'i57', 'type': 'input', 'activation': 'identity'},
             {'name': 'i58', 'type': 'input', 'activation': 'identity'},
             {'name': 'i59', 'type': 'input', 'activation': 'identity'},
             {'name': 'i60', 'type': 'input', 'activation': 'identity'},
             {'name': 'i61', 'type': 'input', 'activation': 'identity'},
             {'name': 'i62', 'type': 'input', 'activation': 'identity'},
             {'name': 'i63', 'type': 'input', 'activation': 'identity'},
             {'name': 'i64', 'type': 'input', 'activation': 'identity'},
             {'name': 'i65', 'type': 'input', 'activation': 'identity'},
             {'name': 'i66', 'type': 'input', 'activation': 'identity'},
             {'name': 'i67', 'type': 'input', 'activation': 'identity'},
             {'name': 'i68', 'type': 'input', 'activation': 'identity'},
             {'name': 'i69', 'type': 'input', 'activation': 'identity'},
             {'name': 'i70', 'type': 'input', 'activation': 'identity'},
             {'name': 'i71', 'type': 'input', 'activation': 'identity'},
             {'name': 'i72', 'type': 'input', 'activation': 'identity'},
             {'name': 'i73', 'type': 'input', 'activation': 'identity'},
             {'name': 'i74', 'type': 'input', 'activation': 'identity'},
             {'name': 'i75', 'type': 'input', 'activation': 'identity'},
             {'name': 'i76', 'type': 'input', 'activation': 'identity'},
             {'name': 'i77', 'type': 'input', 'activation': 'identity'},
             {'name': 'i78', 'type': 'input', 'activation': 'identity'},
             {'name': 'i79', 'type': 'input', 'activation': 'identity'},
             {'name': 'i80', 'type': 'input', 'activation': 'identity'},
             {'name': 'i81', 'type': 'input', 'activation': 'identity'},
             {'name': 'i82', 'type': 'input', 'activation': 'identity'},
             {'name': 'i83', 'type': 'input', 'activation': 'identity'},
             {'name': 'i84', 'type': 'input', 'activation': 'identity'},
             {'name': 'i85', 'type': 'input', 'activation': 'identity'},
             {'name': 'i86', 'type': 'input', 'activation': 'identity'},
             {'name': 'i87', 'type': 'input', 'activation': 'identity'},
             {'name': 'i88', 'type': 'input', 'activation': 'identity'},
             {'name': 'i89', 'type': 'input', 'activation': 'identity'},
             {'name': 'i90', 'type': 'input', 'activation': 'identity'},
             {'name': 'i91', 'type': 'input', 'activation': 'identity'},
             {'name': 'i92', 'type': 'input', 'activation': 'identity'},
             {'name': 'i93', 'type': 'input', 'activation': 'identity'},
             {'name': 'i94', 'type': 'input', 'activation': 'identity'},
             {'name': 'i95', 'type': 'input', 'activation': 'identity'},
             {'name': 'i96', 'type': 'input', 'activation': 'identity'},
             {'name': 'i97', 'type': 'input', 'activation': 'identity'},
             {'name': 'i98', 'type': 'input', 'activation': 'identity'},
             {'name': 'i99', 'type': 'input', 'activation': 'identity'},
             {'name': 'h0', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h1', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h2', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h3', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h4', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h5', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h6', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h7', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h8', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h9', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h10', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h11', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h12', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h13', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h14', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h15', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h16', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h17', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h18', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h19', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h20', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h21', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h22', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h23', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h24', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h25', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h26', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h27', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h28', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h29', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h30', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h31', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h32', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h33', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h34', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h35', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h36', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h37', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h38', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h39', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h40', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h41', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h42', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h43', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h44', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h45', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h46', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h47', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h48', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h49', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h50', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h51', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h52', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h53', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h54', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h55', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h56', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h57', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h58', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h59', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h60', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h61', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h62', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h63', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h64', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h65', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h66', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h67', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h68', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h69', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h70', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h71', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h72', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h73', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h74', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h75', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h76', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h77', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h78', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h79', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h80', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h81', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h82', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h83', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h84', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h85', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h86', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h87', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h88', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h89', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h90', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h91', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h92', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h93', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h94', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h95', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h96', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h97', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h98', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             {'name': 'h99', 'type': 'hidden', 'activation': 'identity',
              'function': 'i0*w+i1*w+i2*w+i3*w+i4*w+i5*w+i6*w+i7*w+i8*w+i9*w+i10*w+i11*w+i12*w+i13*w+i14*w+i15*w+i16*w+i17*w+i18*w+i19*w+i20*w+i21*w+i22*w+i23*w+i24*w+i25*w+i26*w+i27*w+i28*w+i29*w+i30*w+i31*w+i32*w+i33*w+i34*w+i35*w+i36*w+i37*w+i38*w+i39*w+i40*w+i41*w+i42*w+i43*w+i44*w+i45*w+i46*w+i47*w+i48*w+i49*w+i50*w+i51*w+i52*w+i53*w+i54*w+i55*w+i56*w+i57*w+i58*w+i59*w+i60*w+i61*w+i62*w+i63*w+i64*w+i65*w+i66*w+i67*w+i68*w+i69*w+i70*w+i71*w+i72*w+i73*w+i74*w+i75*w+i76*w+i77*w+i78*w+i79*w+i80*w+i81*w+i82*w+i83*w+i84*w+i85*w+i86*w+i87*w+i88*w+i89*w+i90*w+i91*w+i92*w+i93*w+i94*w+i95*w+i96*w+i97*w+i98*w+i99*w+b'},
             ])
n1.start()

print('Symbols:', n1.symbols)
for c_n, c_a in n1.cell_library.items():
    print(c_n, c_a)
for d_n, d_a in n1.diff_parameters.items():
    print(d_n, d_a)

