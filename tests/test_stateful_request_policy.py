import unittest

from ds41f_mlx.serving.request_policy import user_stop_present, validate_stateful_chat_request_policy


class StatefulRequestPolicyTests(unittest.TestCase):
    def test_stateful_stop_policy_allows_absent_null_and_empty_defaults(self):
        for body in (
            b'{"model":"deepseek-v4.1-flash","messages":[]}',
            b'{"stop":null}',
            b'{"stop":""}',
            b'{"stop":[]}',
            {},
        ):
            with self.subTest(body=body):
                self.assertFalse(user_stop_present(body))
                validate_stateful_chat_request_policy(body)

    def test_stateful_stop_policy_rejects_string_and_list_before_mutation(self):
        for body in (b'{"stop":"END"}', b'{"stop":["END"]}', {"stop": "END"}):
            with self.subTest(body=body):
                self.assertTrue(user_stop_present(body))
                with self.assertRaisesRegex(ValueError, "do not support request stop strings"):
                    validate_stateful_chat_request_policy(body)


if __name__ == "__main__":
    unittest.main()
