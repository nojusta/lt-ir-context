# Ar „ir“ visada jungia?

Projektas tiria, kaip iš gretimų lietuviškų žodžių atskirti, kada „ir“ yra jungtukas (CCONJ), o kada dalelytė (PART).

## Modelio mokymas

Naudojama bazinė taisyklė „visada jungtukas“ ir logistinė regresija su vienu arba trimis žodžiais iš abiejų pusių. Modeliai mokomi su train, o ši komanda generuoja prognozes dev daliai. Naudojama Python 3.12.

```sh
python -m pip install -r requirements.txt
python model.py
```
