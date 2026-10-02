
echo " "
echo " "
echo "Extract all revisions:"
echo " "
python pdf_revisions.py EFTA00000001.pdf \
  --all \
  --output-dir EFTA00000001-recovered
echo " "

echo " "
echo " "
echo "Extract the oldest revision:"
echo " "
python pdf_revisions.py EFTA00000001.pdf \
  --revision 1 \
  --output EFTA00000001.oldest.pdf
echo " "