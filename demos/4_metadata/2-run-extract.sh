
echo " "
echo " "
echo "Extract the oldest revision:"
echo " "
python pdf_revisions.py pdf-puzzle.pdf \
  --revision 1 \
  --output pdf-puzzle.oldest.pdf
echo " "

echo " "
echo " "
echo "Let's look at the objects within the PDF:"
echo " "
qpdf --show-object=5,0 --raw-stream-data pdf-puzzle.oldest.pdf
echo " "
echo "Use: https://cryptii.com/pipes/ascii85-encoding/"
echo " "

