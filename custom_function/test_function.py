from torch.autograd import Function
class LinearFunction(Function):
    def symbolic(g, self, mat1, mat2, beta, alpha):
        # return g.op("nonentity", mat1, mat2, self, beta_f=beta, alpha_f=alpha)
        return g.op("nonentity", self, mat1, mat2, beta_f=beta, alpha_f=alpha)

    @staticmethod
    def forward(ctx,input,weight,bias=None,beta_f=1.0,alpha_f=1.0):
        ctx.save_for_backward(input,weight,bias)
        ctx.beta=beta_f
        ctx.alpha=alpha_f

        output=input.mm(weight.t())
        if bias is not None:
            output+=bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        input,weight,bias=ctx.saved_variables
        grad_input=grad_weight=grad_bias=None

        if ctx.needs_input_grad[0]:
            grad_input=grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight=grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias=grad_output.sum(0).squeeze(0)
        return grad_input,grad_weight,grad_bias,None,None

