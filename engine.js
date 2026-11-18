(function(root){
const X=[[0,0],[0,1],[1,0],[1,1]],Y=[0,1,1,0],initial=[2,-2,-2,2,-1,-1,3,3,-1];
const sigmoid=z=>1/(1+Math.exp(-z));
function forward(p,x){const h=[sigmoid(x[0]*p[0]+x[1]*p[2]+p[4]),sigmoid(x[0]*p[1]+x[1]*p[3]+p[5])];const z=h[0]*p[6]+h[1]*p[7]+p[8];return {h,z,q:sigmoid(z)}}
function evaluate(p){const g=Array(9).fill(0),pred=[];let loss=0;X.forEach((x,i)=>{const {h,z,q}=forward(p,x);pred.push(q);loss+=(Math.max(z,0)-z*Y[i]+Math.log1p(Math.exp(-Math.abs(z))))/4;const d=(q-Y[i])/4;for(let j=0;j<2;j++){g[6+j]+=d*h[j];const dh=d*p[6+j]*h[j]*(1-h[j]);g[j]+=dh*x[0];g[2+j]+=dh*x[1];g[4+j]+=dh}g[8]+=d});return {loss,g,pred}}
function train(lr=.03,epochs=300){let p=[...initial],m=Array(9).fill(0),v=Array(9).fill(0),history=[];for(let t=0;t<=epochs;t++){const e=evaluate(p);history.push({epoch:t,p:[...p],...e});if(t===epochs)break;for(let j=0;j<9;j++){m[j]=.9*m[j]+.1*e.g[j];v[j]=.999*v[j]+.001*e.g[j]**2;p[j]-=lr*(m[j]/(1-.9**(t+1)))/(Math.sqrt(v[j]/(1-.999**(t+1)))+1e-8)}}return history}
const api={X,Y,initial,forward,evaluate,train};if(typeof module!=='undefined')module.exports=api;root.MLP=api;
})(typeof globalThis!=='undefined'?globalThis:this);
