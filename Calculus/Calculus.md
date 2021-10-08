


(a) 

The true empirical distribution $\textbf{y}$  is a one-hot vector with a 1 for the true outside word o, and 0 everywhere else. So we can write : 

$y_{w}$ = ${\mathbb{1}_{\{w=o\}}}$ 

and then, 

$-\sum_{w} y_{w}\log(\hat{y}) = -y_{0}\log(\hat{y_{0}}) = -\log(\hat{y_{0}}) $ 
<br/>
<br/>
(b)




$\begin{align*}
\frac{\partial}{\partial{v_{c}}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{v_{c}}}[-u_{o}^{T}v_{c} + \log(\sum_{w}\exp(u_{w}^{T}v_{c}))] \\ 
&=  -u_{o} +\sum_{x}\frac{\exp(u_{x}^{T}v_{c})}{\sum_{w}u_{w}^{T}v_{c}}u_{x} \\ 
&= -u_{o} +\sum_{x}P(O=x/C=c)u_{x} \\ 
&= -y^{T}U^{T} + \hat{y}^{T}U^{T} \\ 
&= U^{T}(\hat{y}-y)^{T}
\end{align*}$
<br/><br/>
(c)


$\frac{\partial}{\partial{u_{w}}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U})
= \frac{\partial}{\partial{u_{w}}}[-u_{o}^{T}v_{c} + \log(\sum_{w}\exp(u_{w}^{T}v_{c}))] \\$
$\textrm{if w = o,}$
$\begin{align*}
\frac{\partial}{\partial{u_{w}}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U})
&= -v_{c} +\frac{\exp(u_{o}^{T}v_{c})}{\sum_{w}u_{w}^{T}v_{c}}v_{c} \\ 
&= -v_{c} +P(O=o/C=c)v_{c} \\ 
&= v_{c}(P(O=o/C=c)v_{c} - 1)  \\ 
\end{align*}$
$\textrm{if w}\neq \textrm{o,}$

$\begin{align*}
\frac{\partial}{\partial{u_{w}}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U})
&= 0 +\frac{\exp(u_{w}^{T}v_{c})}{\sum_{w}u_{w}^{T}v_{c}}v_{c} \\ 
&= 0 +P(O=w/C=c)v_{c} \\ 
&= (P(O=w/C=c)v_{c}  \\ 
\end{align*}$

$\textrm{so for any w,}$
$\begin{align*}
\frac{\partial}{\partial{u_{w}}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U})
&= (\hat{y}_{w} -y_{w})v_{c} \\ 
\end{align*}$
<br/><br/>
(d)


$\textrm{Using the shape convention} \quad \frac{\partial}{\partial{U}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U}) \quad \textrm{must be of the shape of U :}$
$\frac{\partial}{\partial{U}}\textbf{J}_{naive-softmax}(\textbf{v}_{c}, o, \textbf{U}) = \begin{pmatrix}
\frac{\partial}{\partial{u_{1}}}\textbf{J} & \frac{\partial}{\partial{u_{2}}}\textbf{J} & ... & \frac{\partial}{\partial{u_{|Vocab|}}}\textbf{J}\\
\end{pmatrix} $ 
<br/><br/>
(e) 


$\begin{align*}
\sigma^{'}(x) &= \frac{e^{-x}}{(1 + e^{-x})^{2}} \\
&= \frac{-1 +1 +e^{-x}}{(1 + e^{-x})(1 + e^{-x})} \\
&= \sigma(x)\frac{-1 +1 +e^{-x}}{1 + e^{-x}} \\
&= \sigma(x)(1-\sigma(x)) \\
\end{align*}$
<br/><br/>
(f)

$\begin{align*}
\frac{\partial}{\partial{v_{c}}}\textbf{J}_{neg-sample}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{v_{c}}}[-\log(\sigma(u_{o}^{T}v_{c})) - \sum_{k}\log(\sigma(-u_{k}^{T}v_{c}))] \\ 
&=  \frac{-u_{o}\sigma(u_{o}^{T}v_{c})[1-\sigma(u_{o}^{T}v_{c})]}{\sigma(u_{o}^{T}v_{c}} - \sum_{k}\frac{-u_{k}\sigma(-u_{k}^{T}v_{c})[1-\sigma(-u_{k}^{T}v_{c})]}{\sigma(-u_{k}^{T}v_{c}} \\ 
&=  -u_{o}[1-\sigma(u_{o}^{T}v_{c})] + \sum_{k}-u_{k}[1-\sigma(-u_{k}^{T}v_{c})] \\ 
\end{align*}$

$\begin{align*}
\frac{\partial}{\partial{u_{o}}}\textbf{J}_{neg-sample}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{u_{k}}}[-\log(\sigma(u_{o}^{T}v_{c})) - \sum_{k}\log(\sigma(-u_{k}^{T}v_{c}))] \quad (\textrm{k} \neq \textrm{o}) \\ 
&=  \frac{-v_{c}\sigma(u_{o}^{T}v_{c})[1-\sigma(u_{o}^{T}v_{c})]}{\sigma(u_{o}^{T}v_{c}} -  0 \\ 
&=  -v_{c}[1-\sigma(u_{o}^{T}v_{c})]  \\ 
\end{align*}$

$\begin{align*}
\frac{\partial}{\partial{u_{k}}}\textbf{J}_{neg-sample}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{u_{k}}}[-\log(\sigma(u_{o}^{T}v_{c})) - \sum_{j}\log(\sigma(-u_{j}^{T}v_{c}))] \quad (\textrm{k} \neq \textrm{o}) \\ 
&= 0 + \frac{v_{c}\sigma(-u_{k}^{T}v_{c})[1-\sigma(-u_{k}^{T}v_{c})]}{\sigma(-u_{k}^{T}v_{c})}  \\ 
&=  v_{c}[1-\sigma(-u_{k}^{T}v_{c})]  \\ 
\end{align*}$


$\textrm{This loss function is much more efficient to compute than the naive-softmax loss beacause we don't need to go through the all vocabulary.}$
<br/><br/>
(g)

$\begin{align*}
\frac{\partial}{\partial{v_{c}}}\textbf{J}_{neg-sample}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{v_{c}}}[-\log(\sigma(u_{o}^{T}v_{c})) - \sum_{j}\log(\sigma(-u_{j}^{T}v_{c}))] \\ 
&=  \frac{-u_{o}\sigma(u_{o}^{T}v_{c})[1-\sigma(u_{o}^{T}v_{c})]}{\sigma(u_{o}^{T}v_{c})} - \sum_{k}\frac{-u_{k}\sigma(-u_{k}^{T}v_{c})[1-\sigma(-u_{k}^{T}v_{c})]}{\sigma(-u_{k}^{T}v_{c}} \\ 
&=  -u_{o}[1-\sigma(u_{o}^{T}v_{c})] + \sum_{k}-u_{k}[1-\sigma(-u_{k}^{T}v_{c})] \\ 
\end{align*}$

$\begin{align*}
\frac{\partial}{\partial{u_{k}}}\textbf{J}_{neg-sample}(\textbf{v}_{c}, o, \textbf{U}) 
&= \frac{\partial}{\partial{v_{c}}}[-\log(\sigma(u_{o}^{T}v_{c})) - \sum_{j}\log(\sigma(-u_{j}^{T}v_{c}))] \\ 
&= 0 + \frac{\partial}{\partial{v_{c}}}[-\sum_{j=l}^{K}\log(\sigma(-u_{j}^{T}v_{c})) - \sum_{i=1}^{l}\log(\sigma(-u_{k}^{T}v_{c}))] \quad \textrm{where l is the number of time we have drawn} \quad u_{k}\\ 
&=  0 + l\frac{v_{c}\sigma(-u_{k}^{T}v_{c})[1-\sigma(-u_{k}^{T}v_{c})]}{\sigma(-u_{k}^{T}v_{c})}\\
&=  l[1-\sigma(-u_{k}^{T}v_{c})]\\ 
\end{align*}$
<br/><br/>
(h)

$\begin{align*}
(i) \quad \frac{\partial}{\partial{\textbf{U}}}\textbf{J}_{skip-gram}(\textbf{v}_{c}, w_{t-m}, ..., w_{t+m}, \textbf{U}) 
&= \sum\limits_{\substack{-m \leq j \leq m \\ j \neq 0}} \frac{\partial}{\partial{\textbf{U}}}\textbf{J}(\textbf{v}_{c}, w_{t+j} \textbf{U}) \\ 
(ii) \quad \frac{\partial}{\partial{v_{c}}}\textbf{J}_{skip-gram}(\textbf{v}_{c}, w_{t-m}, ..., w_{t+m}, \textbf{U}) 
&= \sum\limits_{\substack{-m \leq j \leq m \\ j \neq 0}} \frac{\partial}{\partial{v_{c}}}\textbf{J}(\textbf{v}_{c}, w_{t+j} \textbf{U})\\
(iii) \quad \frac{\partial}{\partial{v_{w}}}\textbf{J}_{skip-gram}(\textbf{v}_{c}, w_{t-m}, ..., w_{t+m}, \textbf{U}) 
&= \sum\limits_{\substack{-m \leq j \leq m \\ j \neq 0}} \frac{\partial}{\partial{v_{w}}}\textbf{J}(\textbf{v}_{c}, w_{t+j} \textbf{U}) \quad (\textrm{w} \neq \textrm{c})\\
&= 0
\end{align*}$

