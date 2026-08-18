SINTER BURDEN CONTROL — V30 DASHBOARD

FILES
-----
app.py
optimizer.py
requirements.txt

DEPLOYMENT
----------
Keep app.py and optimizer.py in the SAME Streamlit/GitHub project folder.

The dashboard imports:
    import optimizer as opt

Therefore optimizer.py must remain beside app.py.

This package includes the updated v30 backend used as the optimizer module.

The mandate logic is:
- IOL_Fines = 8% of total non-fuel burden
- BF_Returns = 17% of total non-fuel burden

The dashboard additions include:
- Moisture %
- BF_returns
- editable O&M cost
- dry-basis composition table
- wet/as-received composition table
- burden kg/t and burden %
- cost Rs/t and cost %
- existing optimizer/manual/alternative workflow retained
