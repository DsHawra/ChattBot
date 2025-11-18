## SUPABASE CONNECTION ##
from supabase import create_client, Client
from dotenv import load_dotenv
import os
load_dotenv()

SUPABASE_URL = "https://gkspyxwilukotqaizvhe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdrc3B5eHdpbHVrb3RxYWl6dmhlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE0ODQzMDEsImV4cCI6MjA3NzA2MDMwMX0.mYEDAs8-ClmPMiGfS3n9X6i4O23BO-aB5jg_IU2veOc"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


