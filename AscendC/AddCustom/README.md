## 算子原型
- ~~仅支持fp16数据类型~~ 更新支持fp16, fp32, int16, int32数据类型

- 仅适配单核情况 

- ~~不支持广播操作~~ 更新支持 1—8 维广播操作

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">2048,2048</td><td align="center">fp16</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">2048,2048</td><td align="center">fp16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">2048,2048</td><td align="center">fp16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>
