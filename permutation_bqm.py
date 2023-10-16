#! /usr/bin/env python3

import argparse
import sys
import json
import sympy as sp
import re
import pyqubo
import gzip


def error_exit(message: str):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def num_int(x):
    if float(x).is_integer():
        return int(x)
    else:
        error_exit(f"{x} is not integer")

def num(x):
    if float(x).is_integer():
        return int(x)
    else:
        return float(x)


def reorder(*args):
    if len(args) == 1:
        a = args[0]
        return min(int(a[0]), int(a[1])), max(int(a[0]), int(a[1]))
    else:
        return min(int(args[0]), int(args[1])), max(int(args[0]), int(args[1]))


def load_json(json_file_name: str) -> dict:
    """load json file. It can be <stdin> or .json/.json.gz

    Args:
        json_file (str): file name of json file

    Returns:
        dict: json is converted into Python dict.
    """
    if json_file_name == "<stdin>":
        file = sys.stdin
    elif json_file_name.lower().endswith(".json"):
        try:
            file = open(json_file_name, "rt")
        except:
            error_exit(f"Cannot open {json_file_name}")
    elif json_file_name.lower().endswith(".json.gz"):
        try:
            file = gzip.open(json_file_name, "rb")
        except:
            error_exit(f"Cannot open {json_file_name}")
    else:
        error_exit("File {json_file_name} must be .json or .json.gz")
    try:
        json_data = json.load(file)
    except:
        error_exit(f"File {json_file_name} is not a JSON file")
    file.close()
    return json_data


def pyqubo_compile(formula):
    model = formula.compile()
    if is_qubo:
        qubo, offset = model.to_qubo()
        linear = []
        quadratic = []
        for k, v in qubo.items():
            if k[0] == k[1]:
                linear.append([k[0], num_int(v)])
            else:
                quadratic.append([*k, num_int(v)])
        return num(offset), linear, quadratic
    else:
        h, J, offset = model.to_ising()
        linear = [[k, num_int(v)] for k, v in h.items()]
        quadratic = [[*k, num_int(v)] for k, v in J.items()]
        return num(offset), linear, quadratic


def sympy_compile(formula):
    offset = 0
    linear = []
    quadratic = []
    for term in formula.expand().args:
        if term.is_number:  # If term is a single nubmer
            offset = num(term)
        elif isinstance(term, Binary) or isinstance(
            term, Spin
        ):  # If term is a single variable
            linear.append([term, 1])
        elif term.is_Mul:  # If term is a multiplication term
            coeff = 1
            var = []
            for literal in term.expand().args:
                if literal.is_number:
                    coeff = literal
                else:
                    var.append(literal)
            if not float(coeff).is_integer():
                error_exit(f"Term {str(term)} has non integer coefficient")
            if len(var) == 1:
                linear.append([var[0], num_int(coeff)])
            elif len(var) == 2:
                quadratic.append([*var, num_int(coeff)])
            else:
                error_exit(f"Wrong term : {term}")
        else:
            error_exit(f"Wrong term : {term}")
    return offset, linear, quadratic


class ppp:
    """Problem based on one_hot permutation problem"""

    def __init__(self, ppp_file):
        ppp_json = load_json(ppp_file)
        self.ppp = ppp_json.get("ppp")
        if self.ppp is None:
            error_exit('Missing key "ppp"')
        self.mval = ppp_json.get("mval")
        self.nval = ppp_json.get("nval")
        if self.nval is None:
            error_exit('Missing key "nval"')
        if self.mval is None:
            self.mval = self.nval
        self.optimal = ppp_json.get("optimal")
        if self.optimal is None:
            self.optimal = 0


class sp_Spin(sp.Symbol):
    zero = -1
    one = +1

    def _eval_power(self, exp):
        if exp % 2 == 0:
            return sp.Integer(1)
        else:
            return self


class sp_Binary(sp.Symbol):
    zero = 0
    one = 1

    def _eval_power(self, _):
        return self


class pq_Spin(pyqubo.Spin):
    zero = -1
    one = +1


class pq_Binary(pyqubo.Binary):
    zero = 0
    one = 1


class BQM:
    def __init__(self):
        self.m, self.n = sp.symbols("m n")

    def index2(self, x0, x1):
        return reorder(self.index(x0), self.index(x1))

    def latex(self):
        offset, linear, quadratic = compile(self.model_and_offset)
        return (
            "quadratic : "
            + str(sp.latex(sum([c * x * y for x, y, c in quadratic])))
            + "\nlinear: "
            + str(sp.latex(sum([c * v for v, c in linear])))
            + "\noffset : "
            + str(offset)
        )

    def linear_list(self):
        return sorted([[self.index(v[0]), v[1]] for v in self.linear])

    def quadratic_list(self):
        return sorted([[*self.index2(v[0], v[1]), v[2]] for v in self.quadratic])

    def qubo_list(self):
        return sorted(
            [[self.index(v[0]), self.index(v[0]), v[1]]
             for v in self.linear]
            + [[*self.index2(v[0], v[1]), v[2]] for v in self.quadratic]
        )

    def all_list_as_dict(self):
        if self.is_qubo:
            return {
                "linear": self.linear_list(),
                "quadratic": self.quadratic_list(),
                "qubo": self.qubo_list(),
            }
        else:
            return {"h": self.linear_list(), "J": self.quadratic_list()}

    def problem(self, ppp_instance, penalty):
        self.penalty = penalty
        self.model_and_offset = self.penalty * self.model_and_offset
        self.offset_formula *= self.penalty
        self.optimal_model_and_offset_formula *= self.penalty
        problem_formula = sum(
            [v[4] * self.P[v[0]][v[1]] * self.P[v[2]][v[3]]
                for v in ppp_instance.ppp]
        )
        self.problem_offset, self.problem_linear, self.problem_quadratic = compile(
            problem_formula
        )
        self.model_and_offset = problem_formula + self.model_and_offset
        self.offset_formula += self.problem_offset
        if ppp_instance.optimal is not None:
            self.optimal_model_and_offset_formula += (
                self.problem_factor * ppp_instance.optimal
            )

    def to_json_dic(self, problem_name: str):
        self.offset, self.linear, self.quadratic = compile(
            self.model_and_offset)
        if self.mval == self.nval:
            self.nbit = self.nbit.subs({self.m: self.n})
            self.optimal_model_and_offset_formula = (
                self.optimal_model_and_offset_formula.subs({self.m: self.n})
            )
            self.offset_formula = self.offset_formula.subs({self.m: self.n})
            self.linear_term_count = self.linear_term_count.subs(
                {self.m: self.n})
            self.quadratic_term_count = self.quadratic_term_count.subs({
                                                                       self.m: self.n})
        self.optimal_model_formula = (
            self.optimal_model_and_offset_formula - self.offset_formula
        )
        json_dic = {}
        json_dic["problem"] = problem_name
        json_dic["nbit"] = int(self.nbit.subs(
            {self.m: self.mval, self.n: self.nval}))
        json_dic["base"] = 0
        if hasattr(self, "mval") or self.mval == 1 or self.mval == self.nval:
            json_dic["mval"] = self.mval
        json_dic["nval"] = self.nval
        if hasattr(self, "lambda_val"):
            json_dic["lambda"] = self.lambda_val
        json_dic["optimal_model"] = num(
            self.optimal_model_formula.subs(
                {self.m: self.mval, self.n: self.nval})
        )
        json_dic["offset"] = self.offset
        json_dic["offset_computed"] = num(
            self.offset_formula.subs({self.m: self.mval, self.n: self.nval})
        )
        if hasattr(self, "problem_offset"):
            json_dic["problem_offset"] = self.problem_offset
        json_dic["optimal_model_and_offset"] = num(
            self.optimal_model_and_offset_formula.subs(
                {self.m: self.mval, self.n: self.nval}
            )
        )
        json_dic["linear_term_count"] = len(self.linear)
        json_dic["linear_term_count_computed"] = num(
            self.linear_term_count.subs({self.m: self.mval, self.n: self.nval})
        )
        if hasattr(self, "problem_linear"):
            json_dic["problem_linear_count"] = len(self.problem_linear)
        json_dic["quadratic_term_count"] = len(self.quadratic)
        json_dic["quadratic_term_count_computed"] = num(
            self.quadratic_term_count.subs(
                {self.m: self.mval, self.n: self.nval})
        )
        if hasattr(self, "problem_quadratic"):
            json_dic["problem_quadratic_count"] = len(self.problem_quadratic)
        json_dic["linear_terms"] = sorted(
            list({term[1] for term in self.linear}))
        json_dic["quadratic_terms"] = sorted(
            list({term[2] for term in self.quadratic}))
        json_dic["formula"] = {
            "nbit": str(self.nbit.expand()),
            "optimal_model": str(self.optimal_model_formula.expand()),
            "offset": str(self.offset_formula.expand()),
            "optimal_model_and_offset": str(
                self.optimal_model_and_offset_formula.expand()
            ),
            "linear_term_count": str(self.linear_term_count.expand()),
            "quadratic_term_count": str(self.quadratic_term_count.expand()),
        }
        json_dic.update(self.all_list_as_dict())
        return json_dic


class vector(BQM):
    def index(self, literal):
        match = re.match(r"x_{(\d+)}", str(literal))
        if match:
            return int(match.group(1))
        else:
            error_exit(f"Wrong variable: {literal}")

    def __init__(self, nval, is_qubo, is_zero, is_domain):
        super().__init__()
        self.mval, self.nval, self.is_qubo, self.is_zero, self.is_domain = (
            1,
            nval,
            is_qubo,
            is_zero,
            is_domain,
        )
        if self.is_qubo:
            self.X = [Binary(f"x_{{{i}}}") for i in range(self.nval)]
            if is_domain:
                self.X += [sp.Integer(0), sp.Integer(1)]
        else:
            self.X = [Spin(f"x_{{{i}}}") for i in range(self.nval)]
            if self.is_domain:
                self.X += [-sp.Integer(1), sp.Integer(1)]
        self.sum_X = sum([self.X[i] for i in range(self.nval)])

    def compute(self):
        self.nbit = self.n
        if self.is_qubo:
            if self.is_domain:
                self.model_and_offset = sp.Rational(1, 2) * sum(
                    [(self.X[i - 1] - self.X[i]) **
                     2 for i in range(self.nval + 1)]
                )
                self.offset_formula = sp.Rational(1, 2)
                self.optimal_model_and_offset_formula = sp.Rational(1, 2)
                self.linear_term_count = self.n - 1
                self.quadratic_term_count = self.n - 1
            elif self.is_zero:
                self.model_and_offset = (
                    sp.Rational(1, 2) * self.sum_X * (self.sum_X - 1)
                )
                self.offset_formula = sp.Integer(0)
                self.optimal_model_and_offset_formula = sp.Integer(0)
                self.linear_term_count = sp.Integer(0)
                self.quadratic_term_count = sp.Rational(
                    1, 2) * self.n * (self.n - 1)
            else:  # one-hot
                self.model_and_offset = (self.sum_X - 1) ** 2
                self.offset_formula = sp.Integer(1)
                self.optimal_model_and_offset_formula = sp.Integer(0)
                self.linear_term_count = self.n
                self.quadratic_term_count = sp.Rational(
                    1, 2) * self.n * (self.n - 1)
        else:  # Ising
            if self.is_domain:
                self.model_and_offset = sp.Rational(1, 2) * sum(
                    [(self.X[i - 1] - self.X[i]) **
                     2 for i in range(self.nval + 1)]
                )
                self.offset_formula = self.n + 1
                self.optimal_model_and_offset_formula = sp.Rational(
                    1, 2) * 2**2
                self.linear_term_count = self.n - 1
                self.quadratic_term_count = self.n - 1
            elif self.is_zero:
                self.model_and_offset = (
                    sp.Rational(1, 2)
                    * (self.sum_X + self.nval)
                    * (self.sum_X + (self.nval - 2))
                )
                self.offset_formula = sp.Rational(1, 2) * (
                    self.n * (self.n - 2) + self.n
                )
                self.optimal_model_and_offset_formula = sp.Integer(0)
                self.linear_term_count = self.n
                self.quadratic_term_count = sp.Rational(
                    1, 2) * self.n * (self.n - 1)
            else:  # one-hot
                self.model_and_offset = (
                    sp.Rational(1, 2) * (self.sum_X + (self.nval - 2)) ** 2
                )
                self.offset_formula = sp.Rational(
                    1, 2) * ((self.n - 2) ** 2 + self.n)
                self.optimal_model_and_offset_formula = sp.Integer(0)
                self.linear_term_count = self.n
                self.quadratic_term_count = sp.Rational(
                    1, 2) * self.n * (self.n - 1)

    def to_json_dic(self):
        if self.is_qubo:
            model = "QUBO"
        else:
            model = "Ising"
        if self.is_domain:
            json_dic = super().to_json_dic(
                f"{model} model for domain-wall vectors")
        elif self.is_zero:
            json_dic = super().to_json_dic(
                f"{model} model for zero-one-hot vectors")
        else:
            json_dic = super().to_json_dic(
                f"{model} model for one-hot vectors")
        return json_dic


class onehot(BQM):
    def __init__(self, var_class):
        super().__init__()
        self.X = [
            [var_class(f"x_{{{i},{j}}}") for j in range(self.nval)]
            for i in range(self.mval)
        ]
        self.nbit = self.m * self.n
        self.XH = [
            sum([self.X[i][j] for j in range(self.nval)]) for i in range(self.mval)
        ]
        self.XV = [
            sum([self.X[i][j] for i in range(self.mval)]) for j in range(self.nval)
        ]

    def index(self, literal):
        match = re.match(r"x_{(\d+),(\d+)}", str(literal))
        i = int(match.group(1))
        j = int(match.group(2))
        if match:
            return i * self.nval + j
        else:
            error_exit(f"Wrong variable: {literal}")


class onehot_qubo(onehot):
    def __init__(self, mval, nval):
        self.mval, self.nval, self.is_qubo = mval, nval, True
        super().__init__(Binary)
        self.P = self.X
        self.problem_factor = 1

    def compute(self):
        if self.mval == self.nval:
            self.model_and_offset = sp.Rational(1, 2) * sum(
                list(map(lambda x: (x - 1) ** 2, self.XH))
                + list(map(lambda x: (x - 1) ** 2, self.XV))
            )
        else:
            self.model_and_offset = sum(
                list(map(lambda x: (x - 1) ** 2, self.XH))
            ) + sp.Rational(1, 2) * sum(list(map(lambda x: x * (x - 1), self.XV)))

        self.offset_formula = self.m
        self.optimal_model_and_offset_formula = sp.Integer(0)
        self.linear_term_count = self.m * self.n
        self.quadratic_term_count = (
            sp.Rational(1, 2) * self.m * self.m * self.n
            + sp.Rational(1, 2) * self.m * self.n * self.n
            - self.m * self.n
        )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "QUBO model for Permutation enconding by Onehot vectors"
        )
        return json_dic


class onehot_ising(onehot):
    def __init__(self, mval, nval):
        self.mval, self.nval, self.is_qubo = mval, nval, False
        super().__init__(Spin)
        self.P = [[self.X[i][j] + 1 for j in range(nval)] for i in range(mval)]
        self.problem_factor = 4

    def compute(self):
        if self.mval == self.nval:
            self.model_and_offset = sp.Rational(1, 2) * sum(
                list(map(lambda x: (x + (self.nval - 2)) ** 2, self.XH))
                + list(map(lambda x: (x + (self.mval - 2)) ** 2, self.XV))
            )
            self.offset_formula = sp.Rational(1, 2) * (
                ((self.n - 2) ** 2 + self.n) * self.m
                + ((self.m - 2) ** 2 + self.m) * self.n
            )
        else:
            self.model_and_offset = sp.Rational(1, 2) * sum(
                list(map(lambda x: (x + (self.nval - 2)) ** 2, self.XH))
                + list(map(lambda x: (x + self.mval) *
                       (x + (self.mval - 2)), self.XV))
            )
            self.offset_formula = sp.Rational(1, 2) * (
                ((self.n - 2) ** 2 + self.n) * self.m
                + ((self.m - 2) * self.m + self.m) * self.n
            )

        self.optimal_model_and_offset_formula = sp.Integer(0)
        self.linear_term_count = self.m * self.n
        self.quadratic_term_count = (
            sp.Rational(1, 2) * self.m * self.m * self.n
            + sp.Rational(1, 2) * self.m * self.n * self.n
            - self.m * self.n
        )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "Ising model for Permutation enconding by Onehot vectors"
        )
        return json_dic


class all_diff(BQM):
    def __init__(self, var_class):
        super().__init__()
        self.A = [
            [var_class(f"a_{{{i},{j}}}") for j in range(self.nval - 1)]
            + [var_class.zero, var_class.one]
            for i in range(self.nval)
        ]
        self.nbit = self.n * (self.n - 1)
        if self.is_extended:
            self.X = [
                [var_class(f"x_{{{i},{j}}}") for j in range(self.nval)]
                for i in range(self.nval)
            ]
            self.nbit += self.n * self.n

    def index(self, literal):
        match = re.match(r"([ax])_{(\d+),(\d+)}", str(literal))
        c, i, j = match.group(1), int(match.group(2)), int(match.group(3))
        if c == "a":
            return i * (self.nval - 1) + j
        elif c == "x":
            return i * self.nval + j + (self.nval - 1) * self.nval
        else:
            error_exit(f"Wrong variable: {literal}")

    def compute(self):
        self.model_and_offset = sp.Rational(1, 2) * sum(
            [
                (self.A[i][j - 1] - self.A[i][j]) ** 2
                for i in range(self.nval)
                for j in range(self.nval)
            ]
        )
        if self.is_qubo:
            self.offset_formula = sp.Rational(1, 2) * self.n
            self.optimal_model_and_offset_formula = sp.Rational(1, 2) * self.n
        else:
            self.offset_formula = self.n * self.n
            self.optimal_model_and_offset_formula = sp.Rational(
                1, 2) * 4 * self.n


class all_diff_qubo(all_diff):
    def __init__(self, nval, is_extended):
        self.nval, self.mval, self.is_extended, self.is_qubo = (
            nval,
            1,
            is_extended,
            True,
        )
        super().__init__(Binary)
        if not self.is_extended:
            self.P = [
                [self.A[i][j - 1] - self.A[i][j] for j in range(self.nval)]
                for i in range(self.nval)
            ]
            self.problem_factor = 1
        else:
            self.P = self.X
            self.problem_factor = 1

    def compute(self):
        super().compute()
        self.model_and_offset = sum(  # Note: sp.Rational(1, 2) cannot be used
            [
                (sum([self.A[i][j]
                 for i in range(self.nval)]) - (self.nval - 1 - j))
                ** 2
                for j in range(self.nval - 1)
            ]
        ) + self.model_and_offset
        self.offset_formula += (
            sp.Rational(1, 3) * self.n**3
            - sp.Rational(1, 2) * self.n**2
            + sp.Rational(1, 6) * self.n
        )
        if not self.is_extended:
            self.optimal_model_and_offset_formula += 0
            self.linear_term_count = (self.n - 2) * self.n
            self.quadratic_term_count = (
                sp.Rational(1, 2) * self.n**3 - sp.Rational(3, 2) * self.n
            )
        else:
            self.model_and_offset = sum(
                [
                    (self.X[i][j] - (self.A[i][j - 1] - self.A[i][j])) ** 2
                    for j in range(self.nval)
                    for i in range(self.nval)
                ]
            ) + self.model_and_offset
            self.offset_formula += self.n
            self.optimal_model_and_offset_formula += 0
            self.linear_term_count = (self.n - 2) * self.n + self.n * self.n
            self.quadratic_term_count = (
                sp.Rational(1, 2) * self.n**3
                + 2 * self.n**2
                - sp.Rational(7, 2) * self.n
            )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "QUBO model for Permutation enconding by All-diff Domain-wall technique"
        )
        return json_dic


class all_diff_ising(all_diff):
    def __init__(self, nval, is_extended):
        self.nval, self.mval, self.is_extended, self.is_qubo = (
            nval,
            1,
            is_extended,
            False,
        )
        super().__init__(Spin)
        if not self.is_extended:
            self.P = [
                [self.A[i][j - 1] - self.A[i][j] for j in range(self.nval)]
                for i in range(self.nval)
            ]
            self.problem_factor = 4
        else:
            self.P = [
                [self.X[i][j] + 1 for j in range(self.nval)] for i in range(self.nval)
            ]
            self.problem_factor = 4

    def compute(self):
        super().compute()
        self.model_and_offset = sp.Rational(1, 2) * sum(
            [
                (
                    sum([self.A[i][j] for i in range(self.nval)])
                    - (self.nval - 2 - j * 2)
                )
                ** 2
                for j in range(self.nval - 1)
            ]
        ) + self.model_and_offset
        self.offset_formula += (
            sp.Rational(1, 6) * self.n**3 - sp.Rational(1, 6) * self.n
        )
        self.optimal_model_and_offset_formula += 0
        if not self.is_extended:
            self.linear_term_count = (self.n - 2 + self.n % 2) * self.n
            self.quadratic_term_count = (
                sp.Rational(1, 2) * self.n**3 - sp.Rational(3, 2) * self.n
            )
        else:
            self.model_and_offset = sum(
                [
                    ((self.X[i][j] + 1) -
                     (self.A[i][j - 1] - self.A[i][j])) ** 2
                    for j in range(self.nval)
                    for i in range(self.nval)
                ]
            ) + self.model_and_offset
            self.offset_formula += 4 * self.n * self.n - 4 * self.n
            self.optimal_model_and_offset_formula += 0
            self.linear_term_count = 2 * self.n * \
                self.n - (4 - self.n % 2) * self.n
            self.quadratic_term_count = (
                sp.Rational(1, 2) * self.n**3
                + 2 * self.n**2
                - sp.Rational(7, 2) * self.n
            )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "Ising model for Permutation enconding by All-diff Domain-wall technique"
        )
        return json_dic


class domain(BQM):
    def __init__(self, var_class):
        super().__init__()
        if self.lambda_val is None:
            self.lambda_val = 1
        self.A = [
            [var_class(f"a_{{{i},{j}}}") for j in range(self.nval - 1)]
            + [var_class.zero, var_class.one]
            for i in range(self.nval)
        ]
        self.B = [
            [var_class(f"b_{{{i},{j}}}") for j in range(self.nval)]
            for i in range(self.mval - 1)
        ]
        self.B.append([var_class.zero for _ in range(self.nval)])
        self.B.append([var_class.one for _ in range(self.nval)])
        self.nbit = self.m * (self.n - 1) + self.n * (self.m - 1)
        if self.is_extended:
            self.X = [
                [var_class(f"x_{{{i},{j}}}") for j in range(self.nval)]
                for i in range(self.mval)
            ]
            self.nbit += self.m * self.n

    def index(self, literal):
        match = re.match(r"([abx])_{(\d+),(\d+)}", str(literal))
        c, i, j = match.group(1), int(match.group(2)), int(match.group(3))
        if c == "a":
            return i * (self.nval - 1) + j
        elif c == "b":
            return j * (self.mval - 1) + i + self.mval * (self.nval - 1)
        elif c == "x":
            return (
                i * self.nval
                + j
                + self.mval * (self.nval - 1)
                + (self.mval - 1) * self.nval
            )
        else:
            error_exit(f"Wrong variable: {literal}")

    def compute(self):
        self.model_and_offset = (
            self.lambda_val
            * sp.Rational(1, 2)
            * sum(
                [
                    (self.A[i][j - 1] - self.A[i][j]) ** 2
                    for i in range(self.mval)
                    for j in range(self.nval)
                ]
                + [
                    (self.B[i - 1][j] - self.B[i][j]) ** 2
                    for i in range(self.mval)
                    for j in range(self.nval)
                ]
            )
        )
        if self.is_qubo:
            self.offset_formula = (
                self.lambda_val * sp.Rational(1, 2) * (self.m + self.n)
            )
            self.optimal_model_and_offset_formula = (
                self.lambda_val * sp.Rational(1, 2) * (self.m + self.n)
            )
        else:
            self.offset_formula = (
                self.lambda_val * sp.Rational(1, 2) * 2 * (2 * self.m * self.n)
            )
            self.optimal_model_and_offset_formula = (
                self.lambda_val * sp.Rational(1, 2) * (4 * self.m + 4 * self.n)
            )


class domain_qubo(domain):
    def __init__(self, mval, nval, lambda_val, is_extended):
        self.mval, self.nval, self.lambda_val, self.is_extended, self.is_qubo = (
            mval,
            nval,
            lambda_val,
            is_extended,
            True,
        )
        super().__init__(Binary)
        if not self.is_extended:
            self.P = [
                [self.A[i][j - 1] - self.A[i][j] for j in range(self.nval)]
                for i in range(self.nval)
            ]
            self.problem_factor = 1
        else:
            self.P = self.X
            self.problem_factor = 1

    def compute(self):
        super().compute()
        if not self.is_extended:
            self.model_and_offset = sp.Rational(1, 2) * sum(
                [
                    (
                        (self.A[i][j - 1] - self.A[i][j])
                        - (self.B[i - 1][j] - self.B[i][j])
                    )
                    ** 2
                    for i in range(self.mval)
                    for j in range(self.nval)
                ]
            ) + self.model_and_offset
            self.offset_formula += sp.Rational(1, 2) * (self.n + self.m) - 1
            self.optimal_model_and_offset_formula += sp.Rational(1, 2) * (
                self.n - self.m
            )
            self.linear_term_count = 2 * self.m * self.n - 2 * self.m - 2 * self.n
            self.quadratic_term_count = (
                6 * self.m * self.n - 6 * self.m - 6 * self.n + 4
            )
        else:  # extended by auxialy variable
            if self.mval == self.nval:
                self.model_and_offset = sp.Rational(1, 2) * sum(
                    [
                        (self.X[i][j] - (self.A[i][j - 1] - self.A[i][j])) ** 2
                        for j in range(self.nval)
                        for i in range(self.mval)
                    ]
                ) + sp.Rational(1, 2) * sum(
                    [
                        (self.X[i][j] - (self.B[i - 1][j] - self.B[i][j])) ** 2
                        for i in range(self.mval)
                        for j in range(self.nval)
                    ]
                ) + self.model_and_offset
                self.offset_formula += self.n
                self.optimal_model_and_offset_formula += 0
                self.linear_term_count = (
                    3 * self.n * self.n - 4 * self.n - 2 * self.n + 2
                )
                self.quadratic_term_count = (
                    6 * self.n * self.n - 4 * self.n - 4 * self.n
                )
            else:
                self.model_and_offset =  sum(
                    [
                        (self.X[i][j] - (self.A[i][j - 1] - self.A[i][j])) ** 2
                        for j in range(self.nval)
                        for i in range(self.mval)
                    ]
                ) + self.model_and_offset
                self.model_and_offset =  sum(
                    [
                       self.X[i][j] * (1 - (self.B[i - 1][j] - self.B[i][j]))
                        for i in range(self.mval)
                        for j in range(self.nval)
                    ]
                ) + self.model_and_offset
                self.offset_formula += self.m
                self.optimal_model_and_offset_formula += 0
                self.linear_term_count = (
                    3 * self.m * self.n - 3 * self.m - 2 * self.n + 1
                )
                self.quadratic_term_count = (
                    6 * self.m * self.n - 4 * self.m - 4 * self.n
                )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "QUBO model for Permutation enconding by Dual-matrix Domain-wall technique"
        )
        return json_dic


class domain_ising(domain):
    def __init__(self, mval, nval, lambda_val, is_extended):
        self.mval, self.nval, self.lambda_val, self.is_extended, self.is_qubo = (
            mval,
            nval,
            lambda_val,
            is_extended,
            False,
        )
        super().__init__(Spin)
        if not self.is_extended:
            self.P = [
                [self.A[i][j - 1] - self.A[i][j] for j in range(self.nval)]
                for i in range(self.nval)
            ]
            self.problem_factor = 4
        else:
            self.P = [
                [self.X[i][j] + 1 for j in range(self.nval)] for i in range(self.mval)
            ]
            self.problem_factor = 4

    def compute(self):
        super().compute()
        if not self.is_extended:
            self.model_and_offset = sp.Rational(1, 2) * sum(
                [
                    (
                        (self.A[i][j - 1] - self.A[i][j])
                        - (self.B[i - 1][j] - self.B[i][j])
                    )
                    ** 2
                    for i in range(self.mval)
                    for j in range(self.nval)
                ]
            ) + self.model_and_offset
            self.offset_formula += 2 * self.m * self.n - 4
            self.optimal_model_and_offset_formula += (
                sp.Rational(1, 2) * 4 * (self.n - self.m)
            )
            self.linear_term_count = 2 * self.m + 2 * self.n
            self.quadratic_term_count = (
                6 * self.m * self.n - 6 * self.m - 6 * self.n + 4
            )
        else:
            if self.mval == self.nval:
                self.model_and_offset = sp.Rational(1, 2) * sum(
                    [
                        ((self.X[i][j] + 1) -
                         (self.A[i][j - 1] - self.A[i][j])) ** 2
                        for j in range(self.nval)
                        for i in range(self.nval)
                    ]
                ) +  sp.Rational(1, 2) * sum(
                    [
                        ((self.X[i][j] + 1) -
                         (self.B[i - 1][j] - self.B[i][j])) ** 2
                        for i in range(self.nval)
                        for j in range(self.nval)
                    ]
                ) + self.model_and_offset
                self.offset_formula += self.n * (self.n - 1) * 4
                self.optimal_model_and_offset_formula += 0
                self.linear_term_count = self.n * self.n + 4 * self.n - 4
                self.quadratic_term_count = (
                    6 * self.n * self.n - 4 * self.n - 4 * self.n
                )
            else:
                self.model_and_offset = sum(
                    [
                        ((self.X[i][j] + 1) -
                         (self.A[i][j - 1] - self.A[i][j])) ** 2
                        for j in range(self.nval)
                        for i in range(self.mval)
                    ]
                ) + self.model_and_offset
                self.model_and_offset = sum(
                    [
                        (self.X[i][j] + 1) *
                        (2 - (self.B[i - 1][j] - self.B[i][j]))
                        for i in range(self.mval)
                        for j in range(self.nval)
                    ]
                ) + self.model_and_offset
                self.offset_formula += 6 * self.m * self.n - 4 * self.m - 2 * self.n
                self.optimal_model_and_offset_formula += 0
                self.linear_term_count = self.m * self.n + 2 * self.m + 2 * self.n
                self.quadratic_term_count = (
                    6 * self.m * self.n - 4 * self.m - 4 * self.n
                )

    def to_json_dic(self):
        json_dic = super().to_json_dic(
            "Ising model for Permutation enconding by Dual-matrix Domain-wall technique"
        )
        return json_dic


def main():
    global Binary, Spin, is_qubo, compile
    parser = argparse.ArgumentParser(
        description="BQM (QUBO/Ising) model generator for permutation and vectors by one-hot/domain-wall encoding"
    )
    parser.add_argument(
        "-v",
        "--vector",
        action="store_true",
        help="Output QUBO/Ising models for n-bit/qubit vector",
    )
    parser.add_argument(
        "-q", "--qubo", action="store_true", help="Generate QUBO model (default)"
    )
    parser.add_argument(
        "-i", "--ising", action="store_true", help="Generate Ising model"
    )
    parser.add_argument(
        "-z", "--zero", action="store_true", help="Use zero-onehot encoding"
    )
    parser.add_argument(
        "-d", "--domain", action="store_true", help="Use domain-wall encoding"
    )
    parser.add_argument(
        "-a",
        "--all-different",
        action="store_true",
        help="Use all-different technique",
    )
    parser.add_argument(
        "-e", "--extended", action="store_true", help="Use extended auxiliality matrix"
    )
    parser.add_argument(
        "-m",
        "--mval",
        type=int,
        help="Partial permutation selecting m iterms. If omitted m=n is assumed",
    )
    parser.add_argument(
        "-n",
        "--nval",
        type=int,
        help="Partial permutation selecting from n numbers.",
    )
    parser.add_argument(
        "-l",
        "--lambda-val",
        type=int,
        help="Partial permutation selecting from n numbers.",
    )
    parser.add_argument(
        "-p", "--penalty", type=int, help="The value of penalty parameter prioritizing permutation encoding. Must be a positive integer."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Problem file JSON to be written. stdout if omitted",
    )
    parser.add_argument("--pyqubo", action="store_true", help="Use pyqubo for compiling the model (default).")
    parser.add_argument("--sympy", action="store_true", help="Use sympy for compiling the model.")
    parser.add_argument("-t", "--tex", action="store_true",
                        help="output latex formula")
    parser.add_argument(
        "ppp",
        nargs="?",
        type=str,
        default="<stdin>",
        help="OneHot Problem JSON file, <stdin> if omitted.",
    )

    args = parser.parse_args()
    use_pyqubo = args.pyqubo or not args.sympy

    if use_pyqubo:
        (Binary, Spin) = (pq_Binary, pq_Spin)
        compile = pyqubo_compile
    else:
        (Binary, Spin) = (sp_Binary, sp_Spin)
        compile = sympy_compile

    is_vector = args.vector
    is_zero = args.zero
    is_domain = args.domain
    is_qubo = not args.ising
    is_tex = args.tex
    is_all_diff = args.all_different
    is_extended = args.extended
    if is_extended:
        is_domain = True
    lambda_val = args.lambda_val
    penalty = args.penalty
    output = args.output
    mval = args.mval
    nval = args.nval
    ppp_file = args.ppp
    if mval is None:
        mval = nval

    if nval is None:
        ppp_given = True
    else:
        ppp_given = False

    if ppp_given:
        if penalty is None:
            error_exit("Penalty value must be set with the -p option.")
        problem = ppp(ppp_file)
        mval = problem.mval
        nval = problem.nval

    if is_vector:
        model = vector(nval, is_qubo, is_zero, is_domain)
    elif is_all_diff:
        if mval != nval:
            error_exit(f"The values mval({mval}) and nval({nval}) must be equal.")
        if is_qubo:
            model = all_diff_qubo(nval, is_extended)
        else:
            model = all_diff_ising(nval, is_extended)
    elif is_domain:
        if is_qubo:  # dual matrix
            model = domain_qubo(mval, nval, lambda_val, is_extended)
        else:
            model = domain_ising(mval, nval, lambda_val, is_extended)
    else:  # onehot
        if is_qubo:
            model = onehot_qubo(mval, nval)
        else:
            model = onehot_ising(mval, nval)

    model.compute()

    if ppp_given:
        model.problem(problem, penalty)

    if is_tex:
        output_string = model.latex()
    else:
        output_string = re.sub(
            "[\r\n]\s+(-{0,1}\d+)\s*",
            "\\1",
            json.dumps(model.to_json_dic(), sort_keys=False, indent=4),
        )

    print(output_string, file=output)


if __name__ == "__main__":
    main()
