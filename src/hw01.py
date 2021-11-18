####################################################
# CS 5600/6600: F20: Assignment 1
# PALANI JOHNSON
# A02231136
#####################################################

from numpy import array, dot


class percep:
    def __init__(self, weights, delta):
        self.weights = weights
        self.delta = delta

    def output(self, input):
        return 0 if dot(input, self.weights) <= self.delta else 1


class and_percep(percep):
    def __init__(self):
        super().__init__(array([0.51, 0.51]), 1)


class or_percep(percep):
    def __init__(self):
        super().__init__(array([1.01, 1.01]), 1)


class not_percep(percep):
    def __init__(self):
        super().__init__(array([-1.0]), -0.5)


class logic_percep:
    def __init__(self):
        self.OR = lambda x: or_percep().output(x)
        self.AND = lambda x: and_percep().output(x)
        self.NOT = lambda x: not_percep().output(x)


class xor_percep(logic_percep):
    def output(self, x):
        p0 = self.OR(x)
        p1 = self.AND(x)
        p2 = self.NOT(array([p1]))
        return self.AND(array([p0, p2]))


class xor_percep2:
    def __init__(self):
        self.perceps = [
            [
                percep(array([0.51, 0.51]), 1),
                percep(array([1.01, 1.01]), 1),
            ],
            [percep(array([-1.5, 1.01]), 1)],
        ]

    def output(self, input):
        for layer in self.perceps:
            i = []
            for p in layer:
                i.append(p.output(input))
            input = array(i, dtype=float)

        return input


class percep_net(logic_percep):
    def output(self, x):
        p0 = self.OR(array([x[0], x[1]]))
        p1 = self.NOT(array([x[2]]))
        p2 = self.AND(array([p0, p1]))
        return self.OR(array([p2, x[3]]))
