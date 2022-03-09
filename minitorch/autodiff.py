from collections import deque

variable_count = 1


# ## Module 1

# Variable is the main class for autodifferentiation logic for scalars
# and tensors.


class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """

    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0

    def requires_grad_(self, val):
        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            val (bool): whether to require grad
        """
        self.history = History()

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    @property
    def derivative(self):
        return self._derivative

    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def accumulate_derivative(self, val):
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            val (number): value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.zero_derivative_()

    def expand(self, x):
        "Placeholder for tensor variables"
        return x

    # Helper functions for children classes.

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0


# Some helper functions for handling optional tuples.


def wrap_tuple(x):
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


# Classes for Functions.


class Context:
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):  # pragma: no cover
        return self.saved_values


class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError("Need to implement for Task 1.4")
        ans = []
        local_grad = self.last_fn.backward(self.ctx, d_output)
        if not isinstance(local_grad, tuple):
            local_grad = (local_grad,)
        local_grad = list(local_grad)
        for input in self.inputs:
            if not is_constant(input):
                ans.append(local_grad[0])
            local_grad = local_grad[1:]
        return local_grad


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Implement by children class.
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # Tip: Note when implementing this function that
        # cls.backward may return either a value or a tuple.
        # TODO: Implement for Task 1.3.
        # raise NotImplementedError("Need to implement for Task 1.3")
        ans = []
        local_grad = cls.backward(ctx, d_output)
        if not isinstance(local_grad, tuple):
            local_grad = (local_grad,)  # make it a tuple
        local_grad = list(local_grad)
        for input in inputs:
            if not is_constant(input):
                ans.append((input, local_grad[0]))
            local_grad = local_grad[1:]
        return ans


# Algorithms for backpropagation


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def topological_sort(variable):
    """
    Computes the topological order of the computation graph.

    Args:
        variable (:class:`Variable`): The right-most variable

    Returns:
        list of Variables : Non-constant Variables in topological order
                            starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    ans = []
    du = dict()
    du[variable.unique_id] = 1  # starting from right
    # 1. count all edges to calc in-degree
    # 2. do topo sort
    # 1. dfs to calc in-degree
    edges = []

    def dfs(var):
        if var.is_leaf():
            # no edge points to leaf node
            # print(var,var.unique_id)
            return
        else:
            var_input = var.history.inputs
            for input in var_input:
                if not is_constant(input):
                    edges.append((var.unique_id, input.unique_id))
                    # print(var.unique_id,input.unique_id)
                    dfs(input)

    dfs(variable)
    edges = list(set(edges))
    for _, v in edges:
        if v in du.keys():
            du[v] += 1
        else:
            du[v] = 1
    # 2. do topo sort,it assure starting from right-most node
    q = deque()
    du[variable.unique_id] -= 1  # 1 to 0
    q.append(variable)
    while len(q):
        node = q.popleft()
        ans.append(node)
        if node.is_leaf():
            continue
        else:
            inputs = node.history.inputs
            for input in inputs:
                if not is_constant(input):
                    du[input.unique_id] -= 1
                    if du[input.unique_id] == 0:
                        q.append(input)
    return ans


def backpropagate(variable, deriv):
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    See :doc:`backpropagate` for details on the algorithm.

    Args:
        variable (:class:`Variable`): The right-most variable
        deriv (number) : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    # 0. get a topo-sorted queue
    path = topological_sort(variable)
    # 1. create a dictionary of Var and deriv
    d = dict()
    d[variable.unique_id] = deriv
    # 2. pull a Var and deriv
    for p in path:
        # print(p,p.unique_id,p.is_leaf())
        d_out = d[p.unique_id]
        if p.is_leaf():
            p.accumulate_derivative(d_out)
        else:
            last_fn = p.history.last_fn  # has property 'chain_rule'
            ctx = p.history.ctx
            inputs = p.history.inputs
            local_grad = last_fn.chain_rule(ctx, inputs, d_out)
            for var, der in local_grad:
                # print('son',var.unique_id,der)
                if var.unique_id in d.keys():
                    d[var.unique_id] += der
                else:
                    d[var.unique_id] = der
