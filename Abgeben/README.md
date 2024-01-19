# Bat Orientation Calls

Die Teile unseres Projektes (Datapreperation, CNN, FFNN, Trees, CNN+Autoencoder) befinden sich in den jeweiligen Notebooks.

Um die Inhalte der Notebooks ausführen zu können, müssen in jedem Fall mindestens einmal alle Zellen der Sektion **Dateneinlesen, ggf. bearbeiten und abspeichern** aus dem `data_prep.ipynb` Noteboom ausgeführt werden. Diese bereiten die Daten so auf, dass alle anderen Notebooks sie nutzen können.

**Wichtig:** Damit die Daten vom `data_prep.ipynb` Notebook gefunden werden können, müssen sie in folgender Ordnerstruktur angeordnet sein:
```
 - Bat_Orientation_Calls/
    -> Alle Spektogramme
 - Dieser Ordner/
    -> Alle Notebooks
    -> data/
   (-> compressed_pictures/)
 - Auswertung_20220524.csv
 - LMU_20180326_class.csv
 - LMU_20180505_classified.csv
```

Alle benötigten Python-Packages sind in `requirements` zu finden.