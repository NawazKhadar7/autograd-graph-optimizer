"""Pure expression simplifier: constant folding and neutral-element elimination."""
def evaluate(node,values):
    if isinstance(node,(int,float)):return node
    if isinstance(node,str):return values[node]
    op,a,b=node;av,bv=evaluate(a,values),evaluate(b,values)
    if op=='add':return av+bv
    if op=='mul':return av*bv
    raise ValueError('unknown operation')
def simplify(node):
    if not isinstance(node,tuple):return node
    op,a,b=node;a,b=simplify(a),simplify(b)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return evaluate((op,a,b),{})
    if op=='add':
        if a==0:return b
        if b==0:return a
    if op=='mul':
        if a==1:return b
        if b==1:return a
    return (op,a,b)
def node_count(node):return 1 if not isinstance(node,tuple) else 1+node_count(node[1])+node_count(node[2])
